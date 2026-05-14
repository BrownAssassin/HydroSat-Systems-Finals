# HydroSat Systems Presentation Script

This file gives two usable versions of the 7-slide speaking script:

- a short version for about `5-6` minutes
- a fuller version for about `8-9` minutes

Use:

- `Speaker 1`: Arv Bali
- `Speaker 2`: Mrinank S.

## Short Version

### Slide 1 - Cover

- Speaker: `Speaker 1`
- Target time: `40-45s`
- Opening line:
  "Hello everyone, we are HydroSat Systems, and our final-round Track 2 project focuses on on-orbit water-quality inference for turbidity and chlorophyll-a."
- Main script:
  "Our approach is a CPU-first spectral inference pipeline built for the competition's containerized space-computing workflow. We wanted a system that is not just a model, but a portable decision pipeline that can read mounted satellite inputs, run bounded inference, and return compact water-quality products."
- Transition:
  "To explain how that works, I'll hand over to Mrinank for the system architecture."

### Slide 2 - Overall Task Implementation Architecture

- Speaker: `Speaker 2`
- Target time: `45-50s`
- Main script:
  "The implemented baseline follows a simple six-step path. We start from the point table and the Area8 image stack, crop a local 32 by 32 patch around each requested coordinate, generate handcrafted spectral and spatial features, run separate ensemble regressors for turbidity and chlorophyll-a, apply calibration and clipping logic, and finally write the required JSON outputs."
- Transition:
  "The important thing is that this path is deterministic and easy to audit, which matters in a constrained onboard setting. Next, Arv will cover feasibility."

### Slide 3 - On-Orbit Implementation Feasibility

- Speaker: `Speaker 1`
- Target time: `45-50s`
- Main script:
  "Our default runtime path does not require GPU acceleration. The active ensemble bundle is about 29.6 megabytes, and the full checked-in model folder is about 73 megabytes including optional CNN artifacts. On this machine, the full released Area8 offline evaluation processed 475 points in about 20.4 seconds. That makes this a compact baseline that already fits the competition's inference-only container workflow."
- Transition:
  "The next question is what makes this more than a normal ground-only regression workflow."

### Slide 4 - Innovativeness of the Implementation Path

- Speaker: `Speaker 2`
- Target time: `50-55s`
- Main script:
  "The main innovation is not a giant neural network. It is a flight-oriented design choice. We use target-specific water-quality features, keep turbidity and chlorophyll-a as separate tasks, and make the ensemble path the critical path while leaving CNNs optional. That gives us a stronger foundation for future additions like quality gating, uncertainty scoring, regime routing, and selective downlink."
- Transition:
  "Arv will now connect the measured baseline to real application value."

### Slide 5 - Value, Evidence, and Application Scenarios

- Speaker: `Speaker 1`
- Target time: `55-60s`
- Main script:
  "Using the released Area8 truths and the official final-round scoring formula, our current offline baseline produced an algorithm score of 6.05. The score is modest, but it is honest, reproducible, and tied to the actual released evaluation set. The practical value is in the workflow: water utilities, environmental agencies, and emergency teams can use compact water-quality products for rapid triage, while satellite-ground coordination can prioritize downlink for the most decision-relevant cases."
- Transition:
  "That current baseline gives us a floor, and slide six shows how we would improve it."

### Slide 6 - Future Planning

- Speaker: `Speaker 2`
- Target time: `45-50s`
- Main script:
  "Our roadmap has three layers. First, lock the final baseline and evaluation path. Second, improve generalization, especially for turbidity under distribution shift. Third, extend the baseline into a fuller onboard decision system with uncertainty, routing, and selective downlink. So the current repository is already runnable, and the next stage is to make it more robust and more mission-aware."
- Transition:
  "I'll close with the team overview."

### Slide 7 - Team Introduction and Close

- Speaker: `Speaker 1`
- Target time: `35-45s`
- Closing script:
  "HydroSat Systems is a two-person build team covering modeling, evaluation, deployment, and proposal delivery. Arv Bali led the modeling direction and submission ownership, while Mrinank handled evaluation tooling, repo cleanup, and runtime integration. Thank you for your time. HydroSat already provides a real container-ready baseline today, and we believe it can evolve into a stronger onboard water-intelligence system."

## Full Version

### Slide 1 - Cover

- Speaker: `Speaker 1`
- Target time: `65-75s`
- Opening line:
  "Hello everyone, we are HydroSat Systems. Our project for Track 2 focuses on turning multispectral satellite observations into actionable water-quality information in a space-computing environment."
- Main script:
  "Instead of thinking only in terms of model accuracy, we framed the challenge as an onboard decision-system problem. The question becomes: how can a satellite ingest mounted inputs, run bounded inference in a container, and downlink compact turbidity and chlorophyll-a products fast enough to support real decisions? Our answer is a CPU-first spectral inference pipeline that stays compatible with constrained onboard execution while keeping a path open for later neural augmentation."
- Transition:
  "To make that concrete, Mrinank will walk through the end-to-end system architecture."

### Slide 2 - Overall Task Implementation Architecture

- Speaker: `Speaker 2`
- Target time: `75-85s`
- Main script:
  "The architecture is intentionally modular. First, the container reads the mounted point tables and Area8 TIFF image stack. Second, it converts each Lon and Lat request into a local patch, which avoids full-scene processing on the critical path. Third, it computes a large feature set with spectral bands, ratios, water indices, spatial summaries, and seasonal signals. Fourth, it runs target-specific ensemble regressors for turbidity and chlorophyll-a. Fifth, it applies calibration and clipping logic. And finally, it writes the two required Track 2 JSON outputs."
- Extra line:
  "This structure maps directly onto the checked-in modules in the repository, so the implementation path is transparent rather than theoretical."
- Transition:
  "Arv will now explain why that architecture is feasible in an onboard-computing context."

### Slide 3 - On-Orbit Implementation Feasibility

- Speaker: `Speaker 1`
- Target time: `75-85s`
- Main script:
  "We deliberately kept the default inference path CPU-first. The active ensemble bundle is roughly 29.6 megabytes, while the entire checked-in model folder is about 73 megabytes including optional CNN checkpoints. The runtime path does not depend on downloading code or models from outside the container. On this local machine, the full released Area8 evaluation covered 475 points in about 20.4 seconds. That gives us a practical reference for resource consumption, execution latency, and artifact footprint."
- Extra line:
  "In other words, this is not a giant opaque stack that only works in a generous lab environment. It is already a compact, bounded submission package."
- Transition:
  "Now the interesting part is how this differs from a standard ground-only workflow."

### Slide 4 - Innovativeness of the Implementation Path

- Speaker: `Speaker 2`
- Target time: `75-90s`
- Main script:
  "A ground-only pattern often prioritizes shipping entire scenes first and interpreting them later. Our approach is different. We compute point-centered spectral-spatial descriptors that are specific to water-quality inference, we separate turbidity and chlorophyll-a so each task can follow its own feature importance pattern, and we treat CNNs as optional rather than mandatory. That makes the current baseline more robust as a deployment artifact."
- Extra line:
  "It also gives us a clear roadmap for higher-level mission logic: quality gating, optical-regime routing, uncertainty estimation, and selective downlink. Those are the kinds of features that make an onboard system operationally meaningful rather than just computationally impressive."
- Transition:
  "Arv will now connect that design to measured evidence and real application value."

### Slide 5 - Value, Evidence, and Application Scenarios

- Speaker: `Speaker 1`
- Target time: `85-95s`
- Main script:
  "After the organizers released the full Area8 images and truth JSONs, we built an offline evaluator that reconstructs the missing test-point CSVs and scores the repository with the official final-round formula. Our current measured algorithm score is 6.05. That is not yet where we want it, but it is a real benchmark on released data rather than an optimistic proxy. We think that honesty matters."
- Extra lines:
  "The broader value is clear. Water utilities can prioritize sampling after turbidity spikes. Environmental agencies can monitor bloom-like chlorophyll behavior. Emergency response teams can use compact water-quality products for faster triage. And in a satellite-ground workflow, those compact products can be downlinked before heavier imagery."
- Transition:
  "Mrinank will finish the technical story by outlining the future improvement path."

### Slide 6 - Future Planning

- Speaker: `Speaker 2`
- Target time: `70-80s`
- Main script:
  "Our roadmap has three stages. Stage one is to lock the final baseline and preserve a reproducible evaluation path. Stage two is to improve generalization, especially for turbidity under distribution shift, and to strengthen calibration. Stage three is to extend the baseline into a more complete onboard decision system with uncertainty estimates, routing logic, and downlink prioritization. So the current repository is the starting point, not the ceiling."
- Extra line:
  "That balance is important: we already have something that runs, and we also know exactly where the next gains should come from."
- Transition:
  "Arv will close with the team introduction and final takeaway."

### Slide 7 - Team Introduction and Close

- Speaker: `Speaker 1`
- Target time: `55-65s`
- Closing script:
  "HydroSat Systems is a two-person build team. Arv Bali led the modeling direction, submission ownership, and proposal narrative. Mrinank handled evaluation tooling, repo cleanup, environment setup, and runtime integration. Together, our goal was to produce not just a leaderboard attempt, but a container-ready baseline that can credibly evolve into an onboard water-intelligence system."
- Final line:
  "Thank you for listening. We hope HydroSat shows that even a modest but well-structured baseline can be a strong foundation for real space-based water-quality intelligence."
