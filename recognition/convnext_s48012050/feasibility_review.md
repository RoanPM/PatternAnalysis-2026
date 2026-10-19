# Feasibility Review - ConvNeXt on ADNI

## User Need:
The primary user of this data investigation would be medical professionals involved in the diagnosis of Alzheimer’s Disease (AD), with actual patients of AD serving as indirect users. The purpose of this prototype would be to investigate the feasibility ConvNeXt specifically, and machine learning more generally to be actively applied to diagnosis, with a high focus on reducing false negatives.

## Scope:
The scope of this project is to implement the ConvNeXt on the ADNI MRI dataset, as well as using a standard CNN as a baseline to judge improvements both in terms of computation and performance.

## Acceptance Criteria:
- 80% test accuracy
- low false negative rate (below 10%, ideally better)
- training time on the order of hours, not days
- 

The intended user, prototype purpose, and 3–5
testable acceptance criteria addressing the domain’s open dilemma, including an operational/resource
constraint (e.g., peak VRAM, runtime).


## Model Choice:
Hard difficulty, ConvNeXt.

## Course Concepts:
- Convolutions
- Neural Networks & CNN's
- Deep learning

The proposed dataset configuration, model architecture,
chosen difficulty band, and how course concepts and literature support the choices

## Preliminary Feasibility Evidence:


An initial data audit, baseline smoke-test experiment, or
measured resource estimate. A fully trained final model is not required.


## Risks:
- Project scope creep and misallocation of time resources.
- Underestimation of computation runtime
- Unexpected difficulties achieving the acceptance criteria

## Budget:
Computing
- 1-2 per training run on Rangpur, 10+ training hours on home computer (though this will mostly be restricted to testing)

## Fallback:
- The initial CNN as a baseline
- An unoptimised ConvNeXt


Key technical risks, computing budget, the next planned experiment,
and a workable fallback.

