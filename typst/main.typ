#import "@preview/clean-math-paper:0.2.5": *

#let date = datetime.today().display("[month repr:long] [day], [year]")

// Modify some arguments, which can be overwritten in the template call
#page-args.insert("numbering", "1/1")
#text-args-title.insert("size", 2em)
#text-args-title.insert("fill", black)
#text-args-authors.insert("size", 12pt)

#show: template.with(
title: "Jointly predicting RNase H-mediated gapmer potency and tolerability to reduce preclinical screening costs",
authors: (
(name: "Barney Hill", affiliation-id: "1,3,5,*"),
(name: "Nicola Whiffin", affiliation-id: "1,3,4"),
(name: "Stephan J. Sanders", affiliation-id: "1,5,6,7"),
(name: "Carlo Rinaldi", affiliation-id: "1,5,*"),
),
affiliations: (
(id: "1", name: "Department of Paediatrics, University of Oxford, OX3 7TY Oxford, United Kingdom"),
(id: "3", name: "Big Data Institute, University of Oxford, Oxford, UK"),
(id: "4", name: "Broad Center for Mendelian Genomics, Program in Medical and Population Genetics, Broad Institute of MIT and Harvard, Cambridge, MA, USA"),
(id: "5", name: "Institute of Developmental and Regenerative Medicine (IDRM), IMS-Tetsuya Nakamura Building, Old Road Campus, OX3 7TY Oxford, United Kingdom"),
(id: "6", name: "New York Genome Center, New York, NY 10013, USA"),
(id: "7", name: "Department of Psychiatry and Behavioral Sciences, UCSF Weill Institute for Neurosciences, University of California, San Francisco, San Francisco, CA 94178, USA"),
(id: "*", name: "Correspondence to barney.hill@merton.ox.ac.uk, carlo.rinaldi@idrm.ox.ac.uk"),
),
date: date,
link-color: rgb("#008002"),
abstract: [Your abstract goes here.],
)

= Introduction

- Relative clinical success for 2-MOE's in CNS
- Despite this vast majority of amenable conditions remain undeveloped bc of cost constraints.
- ASOs require "thick" preclinical pipelines due to failure rates
- 

= Results

== Fig 1

Summarise data

== Fig 2

Construct preclinical benchmark

== Fig 3

Introduce joint model

= Discussion

// Interpret your findings

= Methods

== Problem Formulation

// Describe your approach

= Code Availability

// Link to code repository if applicable

#bibliography("zotero.bib")
