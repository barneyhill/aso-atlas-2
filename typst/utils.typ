#let R = json("data/paper_numbers.json")

#let comma(n) = {
  let s = str(n)
  let out = ""
  let len = s.len()
  for (i, c) in s.codepoints().enumerate() {
    if i > 0 and calc.rem(len - i, 3) == 0 { out += "," }
    out += c
  }
  out
}

#let fmtdp(val, dp) = {
  let s = str(calc.round(val, digits: dp))
  if dp == 0 { s }
  else {
    let parts = s.split(".")
    let int-part = parts.at(0)
    let frac = if parts.len() > 1 { parts.at(1) } else { "" }
    let pad = range(dp - frac.len()).map(_ => "0").join()
    int-part + "." + frac + pad
  }
}

#let mstd(x) = {
  let ms = str(x.mean)
  let parts = ms.split(".")
  let dp = if parts.len() > 1 { parts.at(1).len() } else { 0 }
  if x.n_folds == none or x.n_folds <= 1 {
    ms
  } else {
    ms + " ± " + fmtdp(x.std, dp)
  }
}
