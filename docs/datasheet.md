# Dataset card

## Purpose

Measure whether an evaluation system distinguishes injected RAG/agent regressions from valid semantic changes and localizes the affected layer.

## Composition

- 100 synthetic scenario groups;
- three trace pairs per group;
- three English-language operational domains;
- one semantic control and two regressions per group;
- group-exclusive development, validation, and challenge splits.

## Generation

`evalmut generate-dataset` deterministically creates the corpus from versioned templates and mutation operators. Public oracle labels record generator intent. They are not substitutes for independent human labels.

## Sensitive information

The corpus is synthetic and contains no real customers, employees, orders, bookings, or proprietary policies.

## Known limitations

Template regularity can make mutations easier to detect. Domain realism and linguistic diversity are limited. Public generation code permits task inspection and contamination. Human calibration is pending for v0.5.

