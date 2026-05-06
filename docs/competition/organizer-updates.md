# Organizer Updates

## Deadline Extension

The organizers extended the `solution proposal` deliverables deadline to **May 15, 2026**.

They also emphasized that the final presentation materials must explicitly address:

- the on-orbit deployment context
- constrained satellite compute and power budgets
- the practical inference path in a space environment

## Test-Set Distribution Warning

The organizers reported a distribution shift between the public training labels and the hidden test labels.

### Released hidden-test summary

Turbidity:

- count: `365`
- min: `0.1`
- median: `1.6`
- average: `2.1874`
- max: `22.8`

Chlorophyll-a:

- count: `103`
- min: `0.18`
- median: `1.42`
- average: `1.6159`
- max: `5.3`

### Practical implication

The local training labels are much heavier-tailed than the hidden test summary, especially for turbidity. That makes distribution-aware sample selection, weighting, calibration, and augmentation likely important once the modeling phase starts.
