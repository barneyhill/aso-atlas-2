import openai
import pydantic
from .config import get_api_key
import json
import time
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import hashlib
import logging
import os
import multiprocessing
import sys
from datetime import datetime
import diskcache

# Fix for macOS: use 'fork' instead of default 'spawn' to avoid subprocess issues
# when running from notebooks or scripts without proper __name__ == '__main__' guards
if sys.platform == 'darwin':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set
from src.prompts import create_xml_to_json_script_prompt, create_helm_script_prompt, create_collate_script_prompt
from src.utils import read_xml, HELMValidator
from src.similarity import find_similar_pairs, resolve_duplicate_chains

def _script_execution_worker(script_text: str, function_name: str, input_data: Any, result_queue: multiprocessing.Queue):
    """
    Worker function to execute scripts in a separate process with isolation.
    """
    try:
        # Re-create isolation environment
        isolated_globals = {
            '__builtins__': __builtins__,
            're': __import__('re'),
            'json': __import__('json'),
            'ET': __import__('xml.etree.ElementTree', fromlist=['ElementTree']),
            'math': __import__('math')
        }
        
        # Clean script logic (duplicated from original _compile_script)
        clean_script = script_text
        if script_text.strip().startswith('{') and "pyscript" in script_text:
            try:
                data = json.loads(script_text)
                if isinstance(data, dict) and 'pyscript' in data:
                    clean_script = data['pyscript']
            except: pass
        clean_script = clean_script.replace('```python', '').replace('```', '').strip()
        
        # Execute script definition
        try:
            # Capture stdout/stderr to prevent random prints from executed scripts
            import sys
            import io
            
            # Redirect stdout to capture prints from the script
            capture_io = io.StringIO()
            original_stdout = sys.stdout
            sys.stdout = capture_io
            
            exec(clean_script, isolated_globals)
            
            if function_name not in isolated_globals:
                sys.stdout = original_stdout  # Restore stdout
                result_queue.put((False, None, f"Function {function_name} not found"))
                return

            # Execute the function
            func = isolated_globals[function_name]
            if isinstance(input_data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in input_data):
                batch_results = []
                all_success = True
                for compound_id, payload in input_data:
                    try:
                        batch_results.append((compound_id, func(payload), ""))
                    except Exception as e:
                        all_success = False
                        batch_results.append((compound_id, None, str(e)))

                sys.stdout = original_stdout  # Restore stdout
                result_queue.put((all_success, batch_results, ""))
                return

            result = func(input_data)
            
            sys.stdout = original_stdout  # Restore stdout
            result_queue.put((True, result, ""))
            
        except Exception as e:
            sys.stdout = original_stdout if 'original_stdout' in locals() else sys.stdout
            result_queue.put((False, None, str(e)))
            
    except Exception as e:
        result_queue.put((False, None, str(e)))

class ASOClient:
    """
    An updated client for OpenAI API calls, tailored for gpt-5 models.
    Refactored to handle script compilation deterministically via string inputs.
    """
    def __init__(self,
                 model: str = "gpt-5",
                 verbosity: str = "low",
                 reasoning_effort: str = "medium",
                 service_tier: str = None,
                 timeout: float = None,
                 max_workers: int = 50,
                 debug: bool = False,
                 skip_no_cache: bool = False,
                 script_timeout: int = 10,
                 api_key: str = None,
                 cache_dir: str = None):

        self.skip_no_cache = skip_no_cache
        self.script_timeout = script_timeout
        self.api_key = api_key or get_api_key('openai')
        base_url = "https://api.openai.com/v1"

        # Auto-adjust timeout for flex processing
        if timeout is None:
            if service_tier == "flex":
                timeout = 900.0  # 15 minutes for flex processing
            else:
                timeout = 600.0  # 10 minutes default

        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.verbosity = verbosity
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.timeout = timeout
        self.max_workers = max_workers
        self.usage = {}
        self.call_count = 0
        self.debug = debug

        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache/responses')
        self.cache = diskcache.Cache(cache_dir)

        # Setup logger if debug enabled
        if self.debug:
            self._setup_logger()

    def __del__(self):
        """Cleanup - close cache on deletion."""
        if hasattr(self, 'cache'):
            self.cache.close()
        # Ensure any lingering processes are cleaned up if we were tracking them, 
        # though process pool executor handles its own workers.

    def _cache_key(self, prompt: str | list, format_spec: any, tools: list,
                   parallel_tool_calls: bool) -> str:
        """Generate a cache key from call parameters."""
        if isinstance(prompt, str):
            prompt_str = prompt
        else:
            prompt_str = json.dumps(prompt, sort_keys=True)

        key_parts = [
            prompt_str,
            self.model,
            self.verbosity,
            self.reasoning_effort,
            str(parallel_tool_calls),
        ]

        if format_spec:
            if isinstance(format_spec, type) and issubclass(format_spec, pydantic.BaseModel):
                key_parts.append(format_spec.__name__)
            elif isinstance(format_spec, dict):
                key_parts.append(json.dumps(format_spec, sort_keys=True))

        if tools:
            key_parts.append(json.dumps(tools, sort_keys=True))

        key_string = '|||'.join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _create_logger(self, name: str, log_filename: str, include_level: bool = True) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers = []

        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)

        format_str = '%(asctime)s - %(levelname)s - %(message)s' if include_level else '%(asctime)s - %(message)s'
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _setup_logger(self):
        os.makedirs('logs', exist_ok=True)
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def _get_file_logger(self, file_path: str, step: str):
        if not self.debug or not hasattr(self, 'session_timestamp'):
            return None
        filename = os.path.basename(file_path).replace('.xml', '')
        log_filename = f'logs/{self.session_timestamp}_{step}_{filename}.log'
        logger_name = f'{step}_{filename}_{self.session_timestamp}'
        file_logger = self._create_logger(logger_name, log_filename, include_level=False)
        file_logger.propagate = False
        return file_logger

    def _close_file_logger(self, file_logger):
        """Close all handlers on a file logger to prevent file handle leaks."""
        if file_logger:
            for handler in file_logger.handlers[:]:
                handler.close()
                file_logger.removeHandler(handler)

    def _log_debug(self, message: str, data: dict = None):
        if self.debug and hasattr(self, 'logger'):
            self.logger.debug(message)
            if data:
                self.logger.debug(f"Data: {json.dumps(data, indent=2, default=str)}")

    def _log_tool_call(self, tool_name: str, args: dict, result: dict, context: str = "", file_logger=None):
        loggers = [self.logger] if self.debug and hasattr(self, 'logger') else []
        if file_logger:
            loggers.append(file_logger)

        for logger in loggers:
            logger.info("")
            logger.info(f"{'='*60}")
            logger.info(f"TOOL EXECUTION: {tool_name} {context}")
            logger.info(f"{'='*60}")
            if result.get('success'):
                logger.info("✓ EXECUTION SUCCESSFUL")
            else:
                logger.info("✗ EXECUTION FAILED")
                logger.info(f"Error: {result.get('error', 'Unknown error')}")
            logger.info(f"{'='*60}\n")

    def _append_tool_result(self, input_messages: list, func_call, tool_result: dict,
                           tool_name: str = None, context: str = "", file_logger=None):
        if tool_name is None:
            tool_name = func_call.name if hasattr(func_call, 'name') else 'unknown'

        args = json.loads(func_call.arguments) if isinstance(func_call.arguments, str) else func_call.arguments
        self._log_tool_call(tool_name, args, tool_result, context=context, file_logger=file_logger)

        input_messages.append({
            "type": "function_call_output",
            "call_id": func_call.call_id,
            "output": json.dumps(tool_result)
        })

    def _handle_api_call_with_retry(self, api_call_func, max_retries: int = 3):
        for attempt in range(max_retries + 1):
            try:
                return api_call_func()
            except openai.APIError as e:
                if e.status_code == 429 and "Resource Unavailable" in str(e) and attempt < max_retries:
                    wait_time = (2 ** attempt) * 60
                    print(f"Flex processing resources unavailable. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif e.status_code == 408 and attempt < max_retries:
                    wait_time = (2 ** attempt) * 30
                    print(f"Request timeout. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        
    def call_model(self,
                   prompt: str | list,
                   format_spec: any = None,
                   tools: list = None,
                   parallel_tool_calls: bool = False,
                   override_service_tier: str = None,
                   metadata: dict = None,
                   use_cache: bool = True):
        
        if use_cache:
            cache_key = self._cache_key(prompt, format_spec, tools, parallel_tool_calls)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            if self.skip_no_cache:
                return None

        def make_api_call():
            if isinstance(prompt, str):
                input_messages = [{"role": "user", "content": prompt}]
            else:
                input_messages = prompt

            params = {
                "model": self.model,
                "input": input_messages,
            }

            if metadata:
                params["metadata"] = metadata
            
            service_tier = override_service_tier or self.service_tier
            if service_tier:
                params["service_tier"] = service_tier
            
            if self.reasoning_effort != "medium":
                params["reasoning"] = {"effort": self.reasoning_effort}

            text_param = {}
            if format_spec and isinstance(format_spec, dict):
                text_param = format_spec
            if self.verbosity != "medium":
                text_param["verbosity"] = self.verbosity
            
            if text_param:
                params["text"] = text_param

            if tools:
                params["tools"] = tools
                if any(t.get('type') == 'custom' for t in tools):
                    params["parallel_tool_calls"] = False
                else:
                    params["parallel_tool_calls"] = parallel_tool_calls
            
            if format_spec and isinstance(format_spec, type) and issubclass(format_spec, pydantic.BaseModel):
                params["text_format"] = format_spec
                response = self.client.responses.parse(**params)
            else:
                response = self.client.responses.create(**params)

            self._track_usage(response)
            return response

        response = self._handle_api_call_with_retry(make_api_call)

        if use_cache:
            cache_key = self._cache_key(prompt, format_spec, tools, parallel_tool_calls)
            self.cache[cache_key] = response

        return response

    def _track_usage(self, response):
        self.call_count += 1
        if self.model not in self.usage:
            self.usage[self.model] = {'input': 0, 'cached_input': 0, 'output': 0}

        if response.usage:
            cached_tokens = 0
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0)

            regular_input_tokens = response.usage.input_tokens - cached_tokens

            self.usage[self.model]['input'] += regular_input_tokens
            self.usage[self.model]['cached_input'] += cached_tokens
            self.usage[self.model]['output'] += response.usage.output_tokens

    def _execute_script_safely(self, script_text: str, function_name: str, input_data: Any) -> tuple[bool, Any, str]:
        """
        Executes a script in a separate process with a timeout.
        """
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_script_execution_worker,
            args=(script_text, function_name, input_data, queue)
        )
        p.start()
        
        try:
            # Wait for result with timeout
            success, result, error_msg = queue.get(timeout=self.script_timeout)
            p.join(timeout=1)  # Give process a moment to cleanup
            if p.is_alive():
                p.terminate()
                p.join()
            return success, result, error_msg
        except (multiprocessing.queues.Empty, TimeoutError):
            if p.is_alive():
                p.terminate()
                p.join()
            return False, None, f"Script execution timed out after {self.script_timeout} seconds"
        except Exception as e:
            if p.is_alive():
                p.terminate()
                p.join()
            return False, None, f"System error during script execution: {str(e)}"

    def validate_json_script_response(self, script_text: str, xml_content: str) -> tuple[bool, str, str]:
        """Validate JSON script from a raw script string using safe execution."""
        # Use safe execution instead of direct exec
        success, json_output, error_msg = self._execute_script_safely(script_text, 'xml_to_json', xml_content)
        
        if not success:
            return False, "", error_msg

        try:
            parsed = json.loads(json_output)
        except json.JSONDecodeError as e:
            return False, json_output, f"Invalid JSON output: {str(e)}"

        if "entries" not in parsed or not isinstance(parsed["entries"], list):
            return False, json_output, "JSON output missing 'entries' list"

        return True, json_output, ""

    def _compute_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _extract_table_number(self, file_path: str) -> str:
        import re
        filename = file_path.split('/')[-1]
        match = re.search(r'table_(\d+)', filename, re.IGNORECASE)
        if match:
            return match.group(1).lstrip('0') or '0'
        return ''

    def _deduplicate_files(self, files: List[str], similarity_threshold: float = 0.90) -> Tuple[List[str], dict]:
        from collections import defaultdict
        file_to_canonical = {}
        hash_to_canonical = {}

        print("Exact hash deduplication...")
        for file_path in tqdm(files, desc="Hashing files"):
            table_num = self._extract_table_number(file_path)
            content_hash = self._compute_file_hash(file_path)
            key = f"{table_num}:{content_hash}"

            if key not in hash_to_canonical:
                hash_to_canonical[key] = file_path
                file_to_canonical[file_path] = file_path
            else:
                file_to_canonical[file_path] = hash_to_canonical[key]

        canonical_after_hash = list(set(file_to_canonical.values()))
        print(f"{len(files)} → {len(canonical_after_hash)} files after exact dedup")

        print(f"\nSimilarity detection (≥{similarity_threshold:.0%})...")
        similar_pairs = find_similar_pairs(canonical_after_hash, threshold=similarity_threshold, show_progress=True)

        if similar_pairs:
            similarity_map = resolve_duplicate_chains(similar_pairs)
            for file_path in list(file_to_canonical.keys()):
                exact_canon = file_to_canonical[file_path]
                if exact_canon in similarity_map:
                    file_to_canonical[file_path] = similarity_map[exact_canon]

        canonical_files = list(set(file_to_canonical.values()))
        print(f"Final: {len(files)} → {len(canonical_files)} files\n")

        return canonical_files, file_to_canonical

    def _build_compound_data_dict(self, compound_id: str, table_data: dict, table_path: str) -> dict:
        compound_data_dict = {'compound_id': compound_id}
        if isinstance(table_data[table_path]['compound_data'][compound_id], list) and len(table_data[table_path]['compound_data'][compound_id]) > 0:
            compound_data_dict.update(table_data[table_path]['compound_data'][compound_id][0])
        return compound_data_dict

    # ========== TOOL DEFINITIONS ==========

    def get_json_script_tool_definition(self) -> dict:
        return {
            "type": "function",
            "name": "execute_json_script",
            "description": "Executes a Python script to convert XML table data to JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pyscript": {
                        "type": "string",
                        "description": "Complete Python 3.11 script containing xml_to_json function"
                    }
                },
                "required": ["pyscript"],
                "additionalProperties": False
            },
            "strict": True
        }

    def get_helm_script_tool_definition(self) -> dict:
        return {
            "type": "function",
            "name": "execute_helm_script",
            "description": "Executes a Python script on the first 3 compounds and returns batch results for validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pyscript": {
                        "type": "string",
                        "description": "Complete Python 3.11 script containing annotate_helm function"
                    }
                },
                "required": ["pyscript"],
                "additionalProperties": False
            },
            "strict": True
        }

    def execute_json_script_tool(self, pyscript: str, xml_content: str) -> dict:
        """Execute JSON script using direct string compilation."""
        is_valid, json_output, error_msg = self.validate_json_script_response(pyscript, xml_content)

        if not is_valid:
            return {
                "success": False,
                "entries": [],
                "error": error_msg,
                "total_count": 0
            }

        try:
            parsed = json.loads(json_output)
            all_entries = parsed.get("entries", [])
            entries_for_llm = []
            for i, entry in enumerate(all_entries[:5]):
                if i < 2:
                    entries_for_llm.append(entry)
                else:
                    entries_for_llm.append({
                        "compound_id": entry.get("compound_id"),
                        "data": {"_summary": f"{len(entry.get('data', {}))} fields"}
                    })

            return {
                "success": True,
                "entries": entries_for_llm,
                "error": "",
                "total_count": len(all_entries),
                "note": "First 2 entries shown with full data."
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to parse JSON output: {str(e)}"}

    def execute_helm_script_tool(self, pyscript: str,
                                 table_data: dict, table_path: str,
                                 xml_content: str) -> dict:
        """Execute HELM script across all compounds in a table in a single run."""
        available_compound_ids = list(table_data[table_path]['compound_data'].keys())
        
        if not available_compound_ids:
            return {
                "success": False,
                "results": [],
                "error": "No compounds found in table data",
                "total_compounds": 0
            }

        compound_inputs = [
            (compound_id, self._build_compound_data_dict(compound_id, table_data, table_path))
            for compound_id in available_compound_ids
        ]

        success, batch_results, error_msg = self._execute_script_safely(
            pyscript, 'annotate_helm', compound_inputs
        )

        if not success and not isinstance(batch_results, list):
            return {
                "success": False,
                "results": [],
                "error": error_msg or "HELM script failed",
                "total_compounds": len(available_compound_ids)
            }

        results = []
        all_success = success

        if isinstance(batch_results, list):
            for compound_id, helm_output, item_error in batch_results:
                is_valid = item_error == ""
                all_success = all_success and is_valid
                results.append({
                    "compound_id": compound_id,
                    "success": is_valid,
                    "helm_output": helm_output if is_valid else None,
                    "error": item_error if not is_valid else None
                })

        return {
            "success": all_success,
            "results": results,
            "error": error_msg if (error_msg and not all_success) else "",
            "total_compounds": len(available_compound_ids)
        }

    def get_collate_script_tool_definition(self) -> dict:
        return {
            "type": "function",
            "name": "execute_collate_script",
            "description": "Executes a Python script on the first 3 compounds and returns batch results for validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pyscript": {
                        "type": "string",
                        "description": "Complete Python 3.11 script containing map_data function"
                    }
                },
                "required": ["pyscript"],
                "additionalProperties": False
            },
            "strict": True
        }

    def execute_collate_script_tool(self, pyscript: str,
                                    table_data: dict, table_path: str,
                                    schema_data: dict) -> dict:
        """Execute Collate script across all compounds in a table in a single run."""
        available_compound_ids = list(table_data[table_path]['compound_data'].keys())
        
        if not available_compound_ids:
            return {
                "success": False,
                "results": [],
                "error": "No compounds found in table data",
                "total_compounds": 0
            }

        compound_inputs = [
            (compound_id, self._build_compound_data_dict(compound_id, table_data, table_path))
            for compound_id in available_compound_ids
        ]

        success, batch_results, error_msg = self._execute_script_safely(
            pyscript, 'map_data', compound_inputs
        )

        if not success and not isinstance(batch_results, list):
            return {
                "success": False,
                "results": [],
                "error": error_msg or "Collate script failed",
                "total_compounds": len(available_compound_ids)
            }

        results = []
        all_success = success

        if isinstance(batch_results, list):
            for compound_id, collate_output, item_error in batch_results:
                is_valid = item_error == ""
                all_success = all_success and is_valid
                results.append({
                    "compound_id": compound_id,
                    "success": is_valid,
                    "collate_output": collate_output if is_valid else None,
                    "error": item_error if not is_valid else None
                })

        return {
            "success": all_success,
            "results": results,
            "error": error_msg if (error_msg and not all_success) else "",
            "total_compounds": len(available_compound_ids)
        }

    # ========== AGENTIC METHODS ==========

    def agentic_json_conversion(self, input_directory: str, n_files: int = None, max_attempts: int = 5) -> tuple[list, list, dict]:
        files = glob.glob(f"{input_directory}/*.xml")
        files = [f for f in files if not f.endswith('_context.txt')]

        if n_files:
            files = files[:n_files]

        canonical_files, file_to_canonical = self._deduplicate_files(files)

        def process_file_agentic(file_path):
            full_xml = read_xml(file_path)
            # (Truncation logic for prompt same as before)
            lines = full_xml.split('\n')
            table_xml_str = '\n'.join(lines[:150] + ['... (truncated) ...'] + lines[-150:]) if len(lines) > 300 else full_xml
            
            prompt = create_xml_to_json_script_prompt(table_xml_str)
            input_messages = [{"role": "user", "content": prompt}]
            tools = [self.get_json_script_tool_definition()]
            has_made_function_call = False

            for attempt in range(max_attempts):
                try:
                    response = self.call_model(
                        prompt=input_messages,
                        tools=tools,
                        parallel_tool_calls=False,
                        metadata={'custom_id': file_path},
                        use_cache=True
                    )
                    
                    if response is None:
                        return {'success': False, 'custom_id': file_path, 'error': "Skipped (no cache)"}

                    for item in response.output:
                        item_dict = item.model_dump(exclude={'status', 'id'}, exclude_none=True)
                        input_messages.append(item_dict)

                    function_calls = [item for item in response.output if item.type == "function_call"]

                    if not function_calls:
                        if not has_made_function_call:
                             input_messages.append({
                                "type": "message", "role": "user",
                                "content": "ERROR: You must call execute_json_script at least once."
                            })
                             continue
                        
                        # Completion - Extract final script
                        result = self._extract_final_script_from_agentic_response(input_messages, full_xml, file_path)
                        return result

                    for func_call in function_calls:
                        if func_call.name == "execute_json_script":
                            has_made_function_call = True
                            args = json.loads(func_call.arguments)
                            pyscript = args.get("pyscript", "")
                            tool_result = self.execute_json_script_tool(pyscript, full_xml)
                            self._append_tool_result(input_messages, func_call, tool_result, tool_name="execute_json_script")

                except Exception as e:
                    return {'success': False, 'custom_id': file_path, 'error': str(e)}

            return {'success': False, 'custom_id': file_path, 'error': "Max attempts reached"}

        successful_responses = []
        failed_responses = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(executor.map(process_file_agentic, canonical_files), total=len(canonical_files)))

        for result in results:
            if result.get('success'): successful_responses.append(result)
            else: failed_responses.append(result)

        return successful_responses, failed_responses, file_to_canonical

    def _extract_final_script_from_agentic_response(self, input_messages: list, xml_content: str, file_path: str) -> dict:
        pyscript = None
        for msg in reversed(input_messages):
            if isinstance(msg, dict) and msg.get("type") == "function_call" and msg.get("name") == "execute_json_script":
                try:
                    args = json.loads(msg.get("arguments", "{}"))
                    if args.get("pyscript"):
                        pyscript = args.get("pyscript")
                        break
                except: continue

        if not pyscript:
            return {'success': False, 'custom_id': file_path, 'error': "No script found"}

        # Validate strict string
        is_valid, json_output, error_msg = self.validate_json_script_response(pyscript, xml_content)
        
        if not is_valid:
             return {'success': False, 'custom_id': file_path, 'error': error_msg}

        return {'success': True, 'custom_id': file_path, 'json_output': json_output}

    def agentic_helm_conversion(self, table_data: dict, custom_ids_to_annotate: dict, max_attempts: int = 5) -> dict:
        tables_to_process = [t for t in custom_ids_to_annotate.keys() if t in table_data]
        
        def process_table_agentic(table_path):
            table_xml_str = read_xml(table_path)
            compound_ids = custom_ids_to_annotate[table_path]
            helm_results = {cid: None for cid in compound_ids}
            
            prompt = create_helm_script_prompt(table_path, table_data)
            input_messages = [{"role": "user", "content": prompt}]
            tools = [self.get_helm_script_tool_definition()]
            has_made_function_call = False

            for attempt in range(max_attempts):
                try:
                    response = self.call_model(
                        prompt=input_messages,
                        tools=tools,
                        parallel_tool_calls=False,
                        metadata={'custom_id': table_path},
                        use_cache=True
                    )

                    if response is None:
                        break

                    for item in response.output:
                        item_dict = item.model_dump(exclude={'status', 'id'}, exclude_none=True)
                        input_messages.append(item_dict)

                    function_calls = [item for item in response.output if item.type == "function_call"]

                    if not function_calls:
                        if not has_made_function_call:
                            input_messages.append({"type": "message", "role": "user", "content": "ERROR: You must call execute_helm_script."})
                            continue
                        break

                    for func_call in function_calls:
                        if func_call.name == "execute_helm_script":
                            has_made_function_call = True
                            args = json.loads(func_call.arguments)
                            pyscript = args.get("pyscript", "")
                            
                            res = self.execute_helm_script_tool(pyscript, table_data, table_path, table_xml_str)
                            
                            for r in res["results"]:
                                if r["success"] and r["compound_id"] in helm_results:
                                    helm_results[r["compound_id"]] = r["helm_output"]
                            
                            self._append_tool_result(input_messages, func_call, res, tool_name="execute_helm_script")

                except Exception as e:
                    break
            
            return table_path, helm_results

        all_helm_annotations = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(executor.map(process_table_agentic, tables_to_process), total=len(tables_to_process)))

        for table_path, helm_results in results:
            all_helm_annotations[table_path] = helm_results
        return all_helm_annotations

    def test_and_retry_collate_script(self, table_data: dict, schema_data: dict, max_attempts: int = 5) -> dict:
        """
        Generates collate scripts using an agentic workflow with tool usage.
        """
        tables_to_process = list(table_data.keys())
        
        def process_table_agentic(table_path):
            print(f"[START] Processing {table_path}...")
            compound_ids = list(table_data[table_path]['compound_data'].keys())

            # Skip tables with no compounds
            if not compound_ids:
                print(f"[SKIP] No compound IDs found for {table_path}")
                return {
                    'success': True,
                    'table_path': table_path,
                    'collate_results': {}
                }

            collate_results = {cid: None for cid in compound_ids}

            # Setup specific file logger
            file_logger = self._get_file_logger(table_path, "step3_collate")
            if file_logger:
                file_logger.info(f"Processing table: {table_path}")
                file_logger.info(f"Compound IDs: {compound_ids}")
            
            prompt = create_collate_script_prompt(table_path, table_data, schema_data)
            
            if file_logger:
                 file_logger.info("PROMPT:")
                 file_logger.info(prompt)
            
            input_messages = [{"role": "user", "content": prompt}]
            tools = [self.get_collate_script_tool_definition()]
            has_made_function_call = False
            
            for attempt in range(max_attempts):
                try:
                    if file_logger:
                         file_logger.info(f"--- Attempt {attempt + 1} ---")

                    cache_key = self._cache_key(input_messages, None, tools, True)
                    cache_hit = cache_key in self.cache
                    if self.debug:
                        print(f"[CACHE] attempt={attempt+1} hit={cache_hit} key={cache_key[:16]}...")
                    if file_logger:
                        if cache_hit:
                            file_logger.info(f"DISK CACHE HIT: {cache_key}")

                    response = self.call_model(
                        prompt=input_messages,
                        tools=tools,
                        parallel_tool_calls=True,
                        metadata={'custom_id': table_path},
                        use_cache=True
                    )
                    
                    if response is None:
                        print(f"[WARN] No response for {table_path} (attempt {attempt+1})")
                        if file_logger: file_logger.warning("No response from model (skipped/no cache)")
                        break

                    for item in response.output:
                        item_dict = item.model_dump(exclude={'status', 'id'}, exclude_none=True)
                        input_messages.append(item_dict)
                        if file_logger and item.type == 'message':
                            file_logger.info(f"MODEL MESSAGE: {item.content}")
                    
                    function_calls = [item for item in response.output if item.type == "function_call"]
                    
                    if not function_calls:
                        if not has_made_function_call:
                            err_msg = "ERROR: You must call execute_collate_script."
                            input_messages.append({"type": "message", "role": "user", "content": err_msg})
                            if file_logger: file_logger.warning(err_msg)
                            continue
                        break

                    for func_call in function_calls:
                        if func_call.name == "execute_collate_script":
                            has_made_function_call = True
                            args = json.loads(func_call.arguments)
                            pyscript = args.get("pyscript", "")
                            
                            if file_logger:
                                file_logger.info(f"TOOL CALL: execute_collate_script")
                                file_logger.info(f"SCRIPT:\n{pyscript}")

                            res = self.execute_collate_script_tool(
                                pyscript, 
                                table_data, 
                                table_path, 
                                schema_data
                            )
                            
                            if file_logger:
                                file_logger.info(f"TOOL RESULT success={res['success']}")
                                if not res['success']: file_logger.error(f"Tool error: {res.get('error')}")
                                else: file_logger.info(f"Results: {res.get('results')}")

                            for r in res["results"]:
                                if r["success"] and r["compound_id"] in collate_results:
                                    collate_results[r["compound_id"]] = r["collate_output"]
                                elif file_logger and not r["success"]:
                                    file_logger.error(f"Failed for {r['compound_id']}: {r.get('error')}")
                            
                            self._append_tool_result(input_messages, func_call, res, tool_name="execute_collate_script", file_logger=file_logger)

                except Exception as e:
                    print(f"[ERROR] Error processing {table_path}: {str(e)}")
                    if file_logger: file_logger.exception(f"Exception during processing: {e}")
                    break
            
            success = sum(1 for v in collate_results.values() if v is not None) > 0

            print(f"[END] Finished {table_path} - Success: {success}")
            if file_logger: file_logger.info(f"FINISHED. Success: {success}")
            self._close_file_logger(file_logger)
            return {
                'success': success,
                'table_path': table_path,
                'collate_results': collate_results
            }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(executor.map(process_table_agentic, tables_to_process), total=len(tables_to_process)))
        
        collate_data = {}
        for result in results:
            collate_data[result['table_path']] = result['collate_results']
        
        return collate_data

    def calculate_cost(self, is_batch: bool = False) -> dict:
        prices = {
            'gpt-5': {'input': 1.25, 'cached_input': 0.125, 'output': 10.00},
            'gpt-5-mini': {'input': 0.25, 'cached_input': 0.025, 'output': 2.00},
        }
        costs = {}
        total_cost = 0.0
        
        for model, usage in self.usage.items():
            model_key = model.split('-202')[0]
            if model_key in prices:
                p = prices[model_key]
                c = (usage['input'] * p['input'] + usage['cached_input'] * p['cached_input'] + usage['output'] * p['output']) / 1_000_000
                if is_batch or self.service_tier == "flex": c *= 0.5
                costs[model] = c
                total_cost += c
        
        costs['total'] = total_cost
        costs['total_calls'] = self.call_count
        return costs
