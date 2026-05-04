#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, center, center, center, center),
  stroke: none,
  inset: (x: 6pt, y: 4pt),

  table.hline(),
  table.header(
    [*Endpoint*], [*Spearman*], [*R#super[2]*], [*RMSE*], [*n*],
  ),
  table.hline(),

  [Efficacy (single-dose)], [0.474 (0.070)], [0.024 (0.177)], [26.73 (2.53)], [116,311 (1,798 tables)],
  [Dose-response (multi-dose)], [0.633 (0.051)], [0.307 (0.103)], [24.87 (1.86)], [63,715 (1,340 tables)],
  [Potency (log IC#sub[50])], [0.531 (0.056)], [---], [0.56 (0.05)], [7,979 / 13,556],
  [Emax], [0.067 (0.062)], [---], [---], [7,979],

  table.hline(),
)
