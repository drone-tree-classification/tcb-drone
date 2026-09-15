# Urban Tree Mapping Project — Specification

## 1. Overview

A system to detect, classify, and map trees across Tucson, AZ (Sonoran Desert region) using drone-captured imagery. The system identifies individual trees, estimates species and (eventually) health via NIR/NDVI-style vegetation indices, and presents the resulting dataset on a web-based interactive map for a small team (2–5 people) to browse, filter, and correct.

**Team**: 2–5 collaborators, hobby/volunteer project, small budget, local compute available (RTX 4080 GPU).

**Phasing**: The pilot begins **RGB-only** — detection, species classification, and mapping — to validate the core pipeline without the added complexity of a second sensor. Near-IR (700–1000nm) capture and NDVI-based health assessment are added in a later phase once the RGB pipeline is proven (see §11).

**Core loop (RGB pilot)**: Fly drone (RGB) → process imagery into an orthomosaic → detect individual trees → classify species → human review/correction → persist to database → view/filter on map.

**Core loop (post-NIR phase)**: adds NIR capture → RGB/NIR co-registration → per-tree NDVI computation → health category, layered onto the same loop.

## 2. Scope

### V1 (Pilot)
- Single park or small neighborhood, roughly a few dozen to ~200 trees.
- One drone flight, **RGB only**, processed end-to-end through the detection/species pipeline.
- Trees only — no cacti/saguaros or other non-tree flora, even though they're locally iconic. (Data model should not hard-block adding vegetation types later, but V1 detection/classification targets trees.)
- Manual, desk-based human review of all detections before they're treated as canonical.
- No automated health assessment yet — health_category defaults to `Not Assessed` until the NIR phase (§5, §11) is implemented.

### Explicitly deferred / not in V1
- NIR capture and NDVI-based health assessment (added in Phase 2 of the roadmap, §11).
- Whole-town scale processing.
- Automated re-flight change detection (see §9).
- Public-facing access or open contribution.
- Non-tree vegetation (cacti, shrubs).
- Cloud hosting (starts self-hosted; see §8).

## 3. Data Collection

### 3.1 Hardware
- Drone-mounted RGB camera (standard photogrammetry survey pass) — used from the start of the pilot.
- NIR camera (700–1000nm) — owned and available, but **not flown in V1**. It gets introduced in Phase 2 (§11) once the RGB-only pipeline is validated. When it is added, note it's a separately mounted sensor, **not** a factory-aligned multispectral rig — RGB and NIR will come from two independent sensors with different lenses/mounting/timing, requiring post-flight co-registration (§4.2).

### 3.2 Flight operations
- Pilot is responsible for legal operation under FAA Part 107 (remote pilot certificate, flight planning, airspace/no-fly-zone checks, altitude limits). This is an operational requirement, not just a data source — track it as a pre-flight checklist item, not a software concern.
- Flight plan should produce sufficient front/side overlap for orthomosaic stitching (typical photogrammetry guidance: ≥75% front overlap, ≥60% side overlap). V1 flights are RGB-only passes; once NIR is introduced (Phase 2), plan for a matching NIR pass over the same area.
- GPS accuracy: consumer-grade drone GPS (~1–3m, worse under dense canopy) is accepted as good enough for this project. **Implication**: in areas with closely spaced trees, GPS alone will not reliably disambiguate individual trees — detection/clustering must rely primarily on the imagery itself (canopy segmentation), with GPS only as an approximate locator, not a precise one. RTK/PPK is not in scope for V1.
- Privacy: incidental capture of private property (backyards, pools, people) is not treated as a major concern for this project's current scale and audience. Revisit if the map or imagery is ever made public — see §9.

## 4. Processing Pipeline

Runs on local compute (RTX 4080). V1 pipeline is **RGB only**; NIR steps are added in Phase 2 (§4.2, §11).

### 4.1 V1 pipeline (RGB only)

1. **Ingest**: raw RGB image set tagged with flight metadata (date, site boundary, drone/camera model).
2. **Orthomosaic stitching**: build a georeferenced RGB orthomosaic using open-source photogrammetry software (e.g. OpenDroneMap/WebODM), self-hosted, using local GPU/CPU.
3. **Tree detection**: run an object detection/segmentation model over the RGB orthomosaic to find individual tree canopies (bounding box or polygon per tree). Fine-tune an existing detection model (e.g. YOLO-family or Detectron2-based segmentation) on the labeled training data already collected, plus data gathered from the pilot flight. Off-the-shelf models are unlikely to be well-tuned for Sonoran Desert tree canopies from directly overhead, so local fine-tuning is expected to matter.
4. **Deduplication / canopy-overlap resolution** (recommended approach, since none was specified):
   - Within a single flight's stitched orthomosaic, duplicate detections mainly come from the detector firing more than once on the same canopy (not from photo overlap, since stitching already resolves that). Cluster detections whose centroids fall within a distance threshold (tuned to typical canopy radius) and merge them into one candidate tree.
   - Where canopies touch/overlap and the detector emits one blob for two trees, or one detection spans what's visibly two canopies, flag as **ambiguous** and route to human review rather than guessing automatically.
5. **Species classification**: run a species classifier (from existing labeled data, extended over time) against each detection's RGB crop, producing a predicted species + confidence score.
6. **Crop & discard raw data**: generate and retain a lightweight per-tree RGB crop for each detection. The full raw photo set and full-resolution orthomosaic are **not** retained long-term (see §4.3) — only these per-tree outputs and the flight-level orthomosaic reference needed to reproduce them if re-processing is ever needed.
7. **Human review queue**: every new detection (and every ambiguous/merge-flagged case) enters a review queue in the web app before being promoted to a canonical Tree record (§6). In V1, `health_category` on each Detection/Tree is simply `Not Assessed` — there's no NIR signal yet to derive it from.

### 4.2 Phase 2 addition: NIR ingest & health assessment

Once the RGB-only pipeline is validated end-to-end, introduce the NIR camera and extend the pipeline:

- **NIR ingest**: raw NIR image set, captured as a separate pass over the same site, tagged to the same flight.
- **NIR orthomosaic stitching**: build a separate georeferenced NIR orthomosaic alongside the RGB one.
- **Co-registration (RGB ↔ NIR)**: since the two cameras are independent and unaligned, register the NIR orthomosaic to the RGB orthomosaic post-flight using image-registration techniques (feature matching / control points — since both are already georeferenced, this is mostly a refinement step). **This is expected to be the biggest technical risk when this phase starts** — validate it on a small test dataset before relying on it for real health data. If registration accuracy is poor, NDVI values will be spatially misaligned with tree canopies and health data will be unreliable.
- **NDVI computation**: for each detected canopy polygon, compute NDVI (and optionally other indices) from the co-registered RGB+NIR data, aggregated per tree (e.g. mean and standard deviation of NDVI across canopy pixels).
- **Health category derivation**: see §5.
- Existing Tree/Detection records from the RGB-only pilot get backfilled with health data on their *next* flight, not retroactively — there's no NIR data for the original flight unless it's re-flown.

### 4.3 Storage policy
- Keep: per-tree RGB crop (and, from Phase 2 onward, per-tree NDVI crop/value), detection metadata, final orthomosaic(s) (compressed) for reference.
- Don't keep long-term: raw individual drone photos once a flight has been successfully processed into an orthomosaic.
- Rationale: town-scale raw imagery is tens of GB per flight; per-tree crops are a small fraction of that and are what the product actually needs day-to-day.

## 5. Health Assessment Model (Phase 2 — not in V1)

V1 has no automated health signal (no NIR data). Every Detection/Tree carries `health_category = Not Assessed` until this phase is implemented. The model below is the design for when NIR is introduced (§4.2, §11).

Store **both** the raw signal and a derived category, not just one or the other:

- `ndvi_mean`, `ndvi_stddev` (raw, continuous) — enables trend tracking and re-tuning thresholds later without re-processing imagery.
- `health_category` (derived, for display/filtering): `Not Assessed | Healthy | Stressed | Declining | Dead | Unknown`.
  - `Not Assessed` = default in V1 / before a tree has any NIR-derived data.
  - `Unknown` covers cases with insufficient/low-quality NIR data (cloud shadow, registration failure, obstructed canopy) — don't force a health guess when the input data doesn't support one.

**Important caveat**: NDVI baselines for arid Sonoran Desert vegetation are naturally lower than the generic NDVI thresholds used in temperate-climate literature (sparse canopy, drought-adapted species like palo verde and mesquite read differently than a temperate hardwood). Category thresholds must be calibrated against locally ground-truthed examples during the pilot (i.e., pick a handful of visibly healthy vs. visibly stressed/dead trees on-site and use their NDVI values to set initial thresholds), not taken from off-the-shelf NDVI health literature.

## 6. Data Model

Core entities:

- **Flight**: id, date, site boundary, pilot, drone/camera info, processing status, orthomosaic references.
- **Detection**: raw pipeline output — flight_id, geometry (bbox/polygon), centroid lat/lon, confidence, RGB crop ref, predicted species + confidence, `review_status` (pending / approved / rejected-false-positive / merged / split), links to the Tree it was matched against (if any). NDVI crop ref, ndvi_mean/stddev, and predicted health_category are **nullable, populated starting Phase 2** — a V1 Detection simply won't have these set.
- **Tree**: canonical, persisted record — id, current lat/lon, confirmed species, current health_category (defaults `Not Assessed`), status (`active | removed | dead | unconfirmed`), first_seen_flight_id, last_seen_flight_id.
- **HealthObservation**: tree_id, flight_id, date, ndvi_mean/stddev, health_category, source detection_id — one per flight a tree was observed in with NIR data, giving a health history over time. Table exists from V1 but stays empty until Phase 2 flights start populating it.
- **Species**: id, common_name, scientific_name, notes (Sonoran Desert regional list: palo verde, mesquite, desert ironwood, acacia, etc.).
- **User**: id, name, email, role (`admin | contributor | viewer`).
- **AuditLog**: who changed what on a Tree/Detection record, when, before/after values — needed given the correction UI is a core feature, so edits are traceable across a 2–5 person team.

## 7. Update Cadence & Change Detection (flagged as open, addressed with a recommendation)

You hadn't settled on re-flight frequency — recommended default for V1: treat each flight as an independent batch, and **don't** try to auto-detect changes (growth/death/removal) across flights yet. Instead:

- On a new flight over a previously-flown area, match new Detections to existing Tree records by proximity (within a tolerance that accounts for the ~1–3m GPS error).
- Unmatched new detections → candidates for new Tree records (human-reviewed).
- Previously-known trees with no matching detection in the new flight → flagged as candidate removed/dead, **not** auto-marked — a person confirms via the review queue.
- Only build automated change-detection/versioning logic once you're actually re-flying regularly enough for it to be worth the complexity.

## 8. Web Application

### 8.1 Map view
- Interactive map (pins/markers per tree, clustering at zoown-out zoom levels once tree count grows).
- Pin color/icon reflects current `health_category`.
- **Layer toggles**: switch between RGB basemap imagery and canopy outline polygons in V1; add an NDVI heatmap overlay option once Phase 2 (NIR) data exists.

### 8.2 Filter/search
- Filter visible trees by species, health category, date detected/last observed, and review status.

### 8.3 Tree detail panel
- Click a tree → panel showing its RGB crop, predicted vs. confirmed species (with confidence score), and detection provenance (which flight/detection it came from). NDVI value/crop, health category, and health history over time (chart) appear once Phase 2 data exists for that tree — until then the panel shows `Not Assessed`.

### 8.4 Review & correction UI
- Queue of pending detections (new + ambiguous/merge-flagged) for reviewers to work through.
- Actions: approve, edit species, edit health category, mark false positive (not a tree), merge two detections into one tree, split one detection into two trees.
- All actions logged to AuditLog (§6) with the acting user.

## 9. Open Risks & Decisions Explicitly Deferred

Carried forward from the interview rather than silently resolved — revisit these as the pilot progresses:

- **RGB/NIR co-registration accuracy** — biggest technical unknown for Phase 2; test on a small sample before relying on it, before it becomes load-bearing for health data across the whole dataset.
- **NDVI health thresholds** — need local ground-truthing in the Sonoran Desert context; don't trust generic literature values. (Phase 2 concern.)
- **GPS accuracy tradeoff** — accepted 1–3m error; will make dense/adjacent tree disambiguation harder and rely more heavily on imagery-based segmentation than on coordinates.
- **Privacy** — not a current concern given project scale, but if the map or raw imagery is ever made public or the area of coverage grows, revisit whether to restrict published imagery resolution/extent around private property.
- **Re-flight cadence/versioning** — no plan yet; V1 uses manual, human-reviewed reconciliation between flights rather than automated change detection (§7).

## 10. Recommended Tech Stack

Chosen for: solo/small-team maintainability, local GPU compute, small budget, self-hosted-first with a later cloud path, and heavy geospatial/ML requirements.

- **Backend**: Python + FastAPI — shares language with the ML/CV pipeline (PyTorch, OpenCV, rasterio/GDAL), simplifying the path from processing pipeline to API.
- **Database**: PostgreSQL + PostGIS — geospatial indexing/queries (proximity matching, bounding-box map queries) are first-class, and it's free/self-hostable.
- **Object storage**: local filesystem (or self-hosted MinIO for an S3-compatible API) for V1; migrate to actual S3/cloud object storage when moving to cloud hosting.
- **Frontend**: React + MapLibre GL JS — open-source map rendering (avoids Google Maps API costs), supports custom raster/vector overlays needed for the NDVI/layer-toggle requirement.
- **Photogrammetry**: OpenDroneMap/WebODM, self-hosted, using local GPU/CPU for orthomosaic stitching.
- **Detection/classification models**: fine-tune an existing YOLO-family (segmentation-capable) or Detectron2-based model on the existing labeled dataset; runs comfortably on an RTX 4080 for pilot-scale data volumes.
- **Auth**: simple email/password or magic-link auth with role-based access (admin/contributor/viewer) — no need for a full IdP at 2–5 users.
- **Deployment**: Docker Compose, self-hosted initially (reachable via something like Tailscale for the team); migrate the web app + database to a small cloud VPS later while keeping heavy ML/photogrammetry processing on local GPU hardware and syncing results up.

## 11. Phased Roadmap

**Phase 0 — Pilot validation (RGB only)**
Pick pilot site. Fly one RGB-only pass. Confirm Part 107 flight plan.

**Phase 1 — RGB Pipeline MVP**
Orthomosaic stitching (RGB), tree detection fine-tuned on existing labels, dedup clustering, species classification, per-tree crop generation. No NIR/health work yet.

**Phase 2 — Web app MVP**
FastAPI + PostGIS backend, React/MapLibre frontend, self-hosted, basic multi-user auth, map view with filters and tree detail panel. All trees show `health_category = Not Assessed`.

**Phase 3 — Review workflow**
Human review queue, correction UI (approve/edit/merge/split/reject), audit log — all against the RGB-only dataset (species, location, detection quality).

**Phase 4 — Introduce NIR & health assessment**
Fly a matching NIR pass over the pilot site. Validate RGB/NIR co-registration on this real data (§9 top risk for this phase) before trusting it at scale. Add NDVI computation, health category derivation, NDVI threshold calibration against ground-truthed pilot trees, NDVI layer/heatmap in the UI, and health history in the tree detail panel.

**Phase 5 — Scale & iterate**
Expand beyond pilot area, move web app/database to cloud hosting, refine cross-flight matching/change tracking as re-flights accumulate, grow the labeled dataset and retrain models.
