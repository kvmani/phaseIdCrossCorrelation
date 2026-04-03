# Glossary and Notation

## Core terms

**CI**  
Confidence index field from EBSD export, used as one quality signal.

**IQ**  
Image quality field from EBSD export.

**Fit**  
Pattern-fit scalar used in quality acceptance rules.

**Qualified record**  
A pixel/sample that passed the configured quality filters, before optional balancing.

**Selected record**  
A qualified record that remained in the final post-balance dataset used for split assignment.

**Full-scan inference**  
Running prediction on every available pattern in a `.oh5` scan and reconstructing a predicted phase map on the native scan grid.

## Symbols

$n_x, n_y$  
scan width and scan height

$i$  
flattened pixel index

$r_\text{train}, r_\text{val}, r_\text{test}$  
split ratios

$n_\text{target}$  
per-phase balancing target count

$p_k$  
predicted probability for class $k$

$\hat{k}$  
predicted class label
