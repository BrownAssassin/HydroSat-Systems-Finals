# HydroSat Systems Presentation Script

This file gives two usable versions of the 7-slide speaking script:

- a short version for about `5-6` minutes
- a fuller version for about `8-9` minutes

Use:

- `Speaker 1`: Arv Bali
- `Speaker 2`: Mrinank S.

Final frozen evidence to cite consistently:

- Turbidity score: `10.62`
- Chl-a score: `18.54`
- Final algorithm score: `14.58`
- Improvement over prior frozen runtime: `+0.13`

Second-layer research evidence to cite carefully:

- Site-adaptive retrospective algorithm score: `53.95`
- Meaning: monitored-site research using historical local calibration
- Do **not** present this as the frozen leaderboard score

## Short Version

### Slide 1 - Cover

- Speaker: `Speaker 1`
- Target time: `40-45s`
- Opening line:
  "Hello everyone, we are HydroSat Systems, and our Track 2 project focuses on on-orbit water-quality inference for turbidity and chlorophyll-a."
- Main script:
  "We built HydroSat as a compact satellite-ready inference pipeline rather than just a standalone model. The goal was to read mounted inputs, run bounded local inference, and output compact JSON water-quality products that match the competition deployment workflow."
- Transition:
  "Mrinank will walk through the system architecture."

### Slide 2 - Overall Task Implementation Architecture

- Speaker: `Speaker 2`
- Target time: `45-50s`
- Main script:
  "Our final runtime follows a six-step path. We read the point request table and multispectral TIFF scenes, extract a 24 by 24 local patch around each coordinate, compute spectral and spatial descriptors, run separate ensemble regressors for turbidity and chlorophyll-a, apply target-aware runtime calibration, and finally write the required Track 2 JSON outputs."
- Transition:
  "That gives us a deterministic and auditable inference path, which is important in a constrained onboard setting. Arv will cover feasibility next."

### Slide 3 - On-Orbit Implementation Feasibility

- Speaker: `Speaker 1`
- Target time: `45-50s`
- Main script:
  "The frozen runtime is CPU-first, so the critical path does not depend on GPU inference. Our final runtime model bundle is only about 4.8 megabytes, and the full released Area8 offline evaluation of 475 points runs in about 25.2 seconds on this machine. That makes the current submission path compact, portable, and aligned with the competition's inference-only container workflow."
- Transition:
  "The next question is what makes this more than a normal ground-only regression system."

### Slide 4 - Innovativeness of the Implementation Path

- Speaker: `Speaker 2`
- Target time: `50-55s`
- Main script:
  "The key innovation is the two-layer system design. First, we use a compact frozen runtime with water-quality-specific handcrafted spectral features and separate target models. Second, for basins that build up local measurement history, we add a site-adaptive layer that uses repeated-station temporal information to refine monitored-site products over time."
- Transition:
  "Arv will now connect that design to our measured results and practical value."

### Slide 5 - Value, Evidence, and Application Scenarios

- Speaker: `Speaker 1`
- Target time: `55-60s`
- Main script:
  "We keep two evidence layers separate. The frozen leaderboard-comparable runtime scores 14.58 on released Area8, with turbidity at 10.62 and chlorophyll-a at 18.54. Separately, in post-release retrospective date-held-out research, a site-adaptive version reached 53.95 once historical same-site calibration was available. That second number is not a leaderboard claim; it shows why the proposed mission architecture becomes stronger after deployment."
- Transition:
  "Mrinank will finish the technical story by explaining what comes next."

### Slide 6 - Future Planning

- Speaker: `Speaker 2`
- Target time: `45-50s`
- Main script:
  "Our roadmap now starts from a compact frozen baseline and a stronger adaptive direction. The next steps are to productionize the site-adaptive layer with uncertainty-aware fallbacks, test causal forward-only adaptation for near-real-time operations, improve turbidity generalization under geographic shift, and extend the system toward selective downlink."
- Transition:
  "Arv will close with the team introduction and final takeaway."

### Slide 7 - Team Introduction and Close

- Speaker: `Speaker 1`
- Target time: `35-45s`
- Closing script:
  "HydroSat Systems is a two-person build team covering modeling, evaluation, deployment, and proposal delivery. Arv Bali led the modeling direction and final submission strategy, while Mrinank handled evaluation tooling, score-push retraining support, runtime integration, and packaging. Thank you for your time. HydroSat now provides a real container-ready baseline at 14.58 and a stronger site-adaptive monitoring direction that becomes more valuable as local history accumulates."

## Full Version

### Slide 1 - Cover

- Speaker: `Speaker 1`
- Target time: `65-75s`
- Opening line:
  "Hello everyone, we are HydroSat Systems. Our Track 2 project focuses on converting multispectral satellite observations into actionable water-quality information in a space-computing environment."
- Main script:
  "We approached this challenge as an onboard decision-system problem rather than only a pure regression task. The question for us was: how can a satellite ingest mounted point requests, run bounded local inference, and downlink compact turbidity and chlorophyll-a products quickly enough to support real decisions? Our answer is a CPU-first spectral inference pipeline with a compact deployment path and a reproducible evaluation loop."
- Transition:
  "Mrinank will make that concrete by walking through the end-to-end system architecture."

### Slide 2 - Overall Task Implementation Architecture

- Speaker: `Speaker 2`
- Target time: `75-85s`
- Main script:
  "The final architecture is modular and practical. First, the container reads the mounted point tables and the multispectral scene bundle. Second, it extracts a 24 by 24 local patch for each requested coordinate, so we avoid full-scene processing on the critical path. Third, we generate a large feature set with spectral bands, water indices, ratios, spatial summaries, and seasonal metadata. Fourth, we run target-specific ensemble regressors for turbidity and chlorophyll-a. Fifth, we apply target-aware runtime calibration. Finally, we write the required Track 2 JSON outputs."
- Extra line:
  "That structure maps directly to the checked-in repository modules, so the implementation path is transparent rather than theoretical."
- Transition:
  "Arv will now explain why that architecture is feasible in an onboard-computing setting."

### Slide 3 - On-Orbit Implementation Feasibility

- Speaker: `Speaker 1`
- Target time: `75-85s`
- Main script:
  "We deliberately kept the default runtime CPU-first. The frozen runtime model bundle is about 4.8 megabytes, and the default path does not depend on runtime downloads or mandatory CNN inference. On this local machine, the full released Area8 evaluation covers 475 points in about 25.2 seconds. That gives us a practical reference for execution time, artifact footprint, and deployment simplicity."
- Extra line:
  "So even though this is still a competition solution, the runtime structure already reflects the kind of bounded, auditable path we would want in a real satellite-ground workflow."
- Transition:
  "Now the important question is how this differs from a conventional ground-only workflow."

### Slide 4 - Innovativeness of the Implementation Path

- Speaker: `Speaker 2`
- Target time: `75-90s`
- Main script:
  "A ground-first pattern often prioritizes sending full imagery down first and interpreting it later. Our design is different. We compute point-centered spectral-spatial descriptors that are directly relevant to water-quality behavior, we separate turbidity and chlorophyll-a so each target can use its own feature importance pattern, and we keep heavier CNN work off the default path. On top of that deployable baseline, we propose a site-adaptive layer that can use historical same-station measurements once a basin is under active monitoring."
- Extra line:
  "That gives us a clearer path for future onboard intelligence features like uncertainty estimation, quality gating, optical-regime routing, and selective downlink."
- Transition:
  "Arv will now connect that design to measured evidence and real application value."

### Slide 5 - Value, Evidence, and Application Scenarios

- Speaker: `Speaker 1`
- Target time: `85-95s`
- Main script:
  "The strongest evidence in the final repository is that we keep two questions separate. For the frozen competition-style runtime, we use the official released Area8 evaluator and reach a reproducible algorithm score of 14.58. For the stronger monitored-basin concept, post-release retrospective date-held-out research reached 53.95 after adding same-site temporal calibration. The first number is the leaderboard-comparable runtime result; the second shows why the mission design is more ambitious than the frozen benchmark model alone."
- Extra lines:
  "In practical terms, a system like this can help water utilities prioritize field verification after sediment events, help environmental agencies monitor bloom-like chlorophyll behavior, and help mission operators downlink compact decision products before heavier imagery."
- Transition:
  "Mrinank will finish by outlining what the next development stage should focus on."

### Slide 6 - Future Planning

- Speaker: `Speaker 2`
- Target time: `70-80s`
- Main script:
  "Our roadmap now starts from a compact frozen baseline and a stronger adaptive direction. First, we want to productionize the site-adaptive layer with uncertainty-aware fallbacks. Second, we want to test causal forward-only adaptation for near-real-time operations. Third, we want to improve turbidity robustness under geographic and distribution shift, then extend the pipeline toward regime-aware routing and selective downlink."
- Extra line:
  "The important point is that the current repository is already runnable, and the roadmap builds on that instead of replacing it."
- Transition:
  "Arv will close with the team introduction and final takeaway."

### Slide 7 - Team Introduction and Close

- Speaker: `Speaker 1`
- Target time: `55-65s`
- Closing script:
  "HydroSat Systems is a two-person build team. Arv Bali led the modeling direction, submission ownership, and final proposal narrative. Mrinank handled evaluation tooling, late-stage score-push experiment infrastructure, environment setup, runtime integration, and packaging. Together, our goal was to deliver not only a leaderboard attempt, but a compact and credible onboard inference baseline."
- Final line:
  "Thank you for listening. HydroSat now delivers a reproducible frozen runtime at 14.58 and a clearer path toward site-adaptive water-quality intelligence as monitoring history accumulates."
