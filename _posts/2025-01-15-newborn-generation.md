---
title: "Generating Realistic Newborn Images from Ultrasound with Stable Diffusion 1.5"
date: 2025-01-15 00:00:00
description: "Exploring how Stable Diffusion 1.5, ControlNet, Facial landmarks, and Depth conditioning can transform ultrasound images into realistic newborn portraits while preserving facial structure."
permalink: /posts/2025/01/newborn-generation/
---

> **Can we generate a plausible newborn portrait from a 3D/4D ultrasound image while preserving its observable facial structure and head pose?**

![Example ultrasound-to-newborn result](/images/newborn-generation/result.png)
*Ultrasound-to-newborn generation. The output is a plausible visualization, **not** a prediction of the baby's actual future appearance.*

The difficult part was not generating a photorealistic baby face. It was **controlling the geometry** of that face using an input with noise boundaries, missing details, and almost no photographic texture. I explored a structure-guided diffusion pipeline combining **Depth ControlNet**, **FaceMesh ControlNet**, and a **newborn-specific LoRA**. Along the way, I investigated a recurring failure: **closed eyelids were often distorted even when the overall pose looked correct**

| Project at a glance | |
|---|---|
| **Focus** | Generative AI · Computer Vision · Structure-guided image generation |
| **Base model** | Realistic Vision V5.1 (Stable Diffusion 1.5 family) |
| **Key methods** | Depth Anything V2 · ControlNet · MediaPipe FaceMesh · LoRA |
| **Implementation** | Python · PyTorch · Hugging Face Diffusers · OpenCV |
| **My work** | Ultrasound preprocessing, conditioning experiments, LoRA fine-tuning, multi-condition integration, and exploratory evaluation |

## 1. The Challenge: photorealism is not enough

**Input:** a 3D/4D fetal facial ultrasound image.

**Output:** a photorealistic *plausible* newborn portrait.

The image should retain as much **observable structure** as possible: approximate eye and mouth positions, nose shape, coarse facial proportions, and head orientation. This is inherently underconstrained: ultrasound and natural photographs belong to different visual domains, and ultrasound does not reliably reveal fine details such as eyelid boundaries or skin texture.

That creates the central trade-off of this project:

> **Too little structural guidance ->** appealing images that drift from the input.

> **Too much structural guidance ->** stronger alignment, but more facial artifacts.

I treated this as an engineering investigation: finding a conditioning representation that carries *reliable* information, then give the diffusion model enough freedom to synthesize the rest.


## 2. What I built
![System architecture for ultrasound-to-newborn generation](/images/newborn-generation/pipeline.png)
*Overall pipeline: ultrasound preprocessing produces structural conditions, which guide a photorealistic diffusion backbone with a newborn appearance prior.*

Each component has a distinct responsibility:

| Component | Role in the system | What I implemented or investigated |
|---|---|---|
| **Realistic Vision V5.1** | Photorealistic image prior | Baseline comparison against vanilla SD 1.5 |
| **Depth Anything V2 + Depth ControlNet** | Coarse head pose, contour, and nose structure| Ultrasound-specific preprocessing, depth-map postprocessing, scale sweeps, failure analysis |
| **MediaPipe FaceMesh + FaceMesh ControlNet** | Explicit guidance for eye and mouth geometry | Landmark extraction, fallback enhancement, alignment checks, local guidance experiments |
| **Newborn LoRA** | Bias toward sleeping-newborn appearance | Dataset preparation, captioning, training, strength comparison |
| **Multi-ControlNet** | Combine complementary structural cues | Joint conditioning and tuning across examples |
| **Evaluation** | Check realism and observable geometry | CLIP/distributional metrics and landmark-based geometry comparisons |

The distinction matters: **I did not train Stable Diffusion or the ControlNets from scratch.** The work was in adapting existing models to a difficult input domain, designing the conditions, and diagnosing their limitations.


## 3. Building the solution, one failure at a time

### Step 1 — Establish a photorealistic baseline

I first compared vanilla **Stable Diffusion 1.5** with **Realistic Vision 5.1**, keeping the generaion settings consistent. Realistic Vision produced more natural baby facial features, skin texture, and lighting, so I selected it as the generative backbone.

<div style="display: flex; gap: 16px; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; min-width: 0;">
    <img src="/images/newborn-generation/sd.png" alt="Vanilla Stable Diffusion 1.5 baseline" style="width: 100%;">
    <figcaption style="text-align: center; font-size: 0.9em;">Stable Diffusion 1.5</figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; min-width: 0;">
    <img src="/images/newborn-generation/rv.png" alt="Realistic Vision V5.1 baseline" style="width: 100%;">
    <figcaption style="text-align: center; font-size: 0.9em;">Realistic Vision V5.1</figcaption>
  </figure>
</div>

**Decision:** use Realistic Vision for better photorealism. But text-to-image generation alone offers no dependable link to the **source ultrasound's geometry**.

### Step 2 - Can depth preserve the pose?

**Hypothesis:** a relative depth map may capture coarse face geometry well enough to guide head orientation and nose shape - even though it cannot recover automatically accurate depth from ultrasound.

I used **Depth Anything V2 (ViT-B)** as a depth estimator and designed a preprocessing pipeline for ultrasound's dark background and noisy surface texture. 


![Ultrasound-to-depth preprocessing stages](/images/newborn-generation/depth_process.png)
*From ultrasound image to a cleaned relative-depth condition.*

The important design decisions were:

1. **Preserve geometry:** resize with aspect-ratio-preserving padding to 512 × 512 rather than stretch the face.
2. **Separate the foreground:** threshold, morphological closing, and connected-component analysis to focus on the visible facial region. Denoising reduces speckle-like artifacts.
3. **Normalize inside the facial mask:** clip to the 1st–99th percentiles so the large black background does not dominate the depth range.
4. **Keep a distinct silhouette:** map foreground values to `[0.08, 1]`, set the background to `0`, and feather the boundary to avoid an unnaturally sharp edge.

I also tried **CLAHE**; in these experiments it offered little depth improvement and sometimes amplified unwanted surface patterns, so it was not part of the selected depth pipeline.

![Depth ControlNet outputs under different scales](/images/newborn-generation/depth_scale.png)
*Depth-guidance sweep, with other generation settings held constant.*

**What improved:** around a conditioning scale of **0.5**, the output followed the input's coarse head pose and facial structure more clearly.  
**What failed:** eyelids were sometimes doubled, misshapen, or looked partly open. Some faces also appeared older than newborns.

#### The most useful debugging experiment: why did eyelids fail?

Rather than just increasing depth strength, I tested whether **the depth representation contained eyelid information at all**.

![Estimated depth from an RGB sleeping-newborn image](/images/newborn-generation/rgb_depth.png)
*RGB newborn image versus its estimated depth: the nose and facial surface are apparent, but fine closed-eyelid boundaries are weak.*

![Estimated depth from fetal ultrasound](/images/newborn-generation/us_depth.png)
*Ultrasound-derived depth: fine boundaries are even more ambiguous amid missing regions and imaging artifacts.*

Even when estimated from a clean RGB image, relative depth barely captured the eyelid boundary. The ultrasound domain added more noise and missing information. This suggested the problem was **not solved by stronger Depth ControlNet alone**: a condition cannot reliably enforce detail it does not encode.

> **Finding:** depth is useful for *global 3D-like structure*, but weak as an explicit description of closed-eye contours.

**Next decision:** provide a separate, more local representation for the eye region.

### Step 3 — Add explicit facial landmarks

I tested a **MediaPipe FaceMesh → FaceMesh ControlNet** branch to provide clearer positional cues, especially for eyes and mouth.

![FaceMesh processing pipeline](/images/newborn-generation/fm_process.png)
*Landmark extraction, visual alignment checks, and a rendered FaceMesh condition.*

I used the same resize/padding transform as the depth branch to keep both conditions spatially aligned. Because ultrasound is challenging for a face detector, I tried lower detection/presence thresholds (**0.1**) and a **CLAHE fallback** when detection failed; I also overlaid the landmarks on the input for manual alignment verification. Iris connections were omitted because the closed eyes provided no reliable iris information.

![FaceMesh guidance scale comparison](/images/newborn-generation/fm_scale.png)
*FaceMesh guidance controls local geometry, but excessive strength can distort the face.*

| FaceMesh scale | Observation |
|---|---|
| **0.3** | Weak but visible positional guidance |
| **0.5–0.7** | Better eye/mouth alignment with reasonable realism |
| **1.0** | Over-constraint and distortions around facial features |

**Decision:** use **moderate FaceMesh guidance** (0.5 in the single-branch experiments) and emphasize useful local features rather than forcing an entire dense mesh onto an ambiguous face.

### Step 4 — Correct apparent age with a newborn LoRA

A separate problem remained: realistic generated baby faces sometimes looked more like **older infants** than sleeping newborns. Geometry control cannot fix the appearance prior by itself.

I fine-tuned a lightweight **LoRA on approximately 40 face-focused images of Asian newborns**, with image-specific captions, using Realistic Vision V5.1 as the base. The adaptation was intended to favor newborn-specific proportions and a naturally sleeping appearance—not to encode the identity of the fetus in an ultrasound.

| Training setting | Value |
|---|---|
| Images / resolution | ~40 / 512 × 512 |
| LoRA rank | 8 |
| Batch size / gradient accumulation | 2 / 2 (effective 4) |
| Steps / learning rate | 600 / 5 × 10⁻⁵ |
| Scheduler | Constant with 30 warmup steps |
| Precision / augmentation | FP16 / horizontal flip |

![LoRA strength comparison](/images/newborn-generation/lora_scale.png)
*With a fixed seed, increasing LoRA strength shifts the output toward the newborn training distribution.*

A LoRA scale of **0.5** gave a useful balance: softer, more newborn-like features without letting the small training set dominate the output. Higher strength sometimes made results smoother and less varied.

> **Division of labor:** ControlNet guides **where facial structures go**; LoRA influences **what kind of newborn appearance is generated**.

### Step 5 — Combine depth and FaceMesh

The final structure-guided approach paired the two complementary conditions:

- **Depth:** global pose, facial surface, and coarse nose geometry.
- **FaceMesh:** more explicit eye and mouth landmarks.

```python
# Illustration of the two independently tunable conditions
controlnet_conditioning_scale = [depth_scale, facemesh_scale]
control_guidance_start = [0.0, 0.0]
control_guidance_end = [depth_end, facemesh_end]
```

![Depth and FaceMesh joint-conditioning grid](/images/newborn-generation/d_fm_scale.png)
*A joint scale sweep: increasing both guidance strengths is not automatically better.*

The experiments exposed another trade-off: strong simultaneous guidance can **over-constrain** the generation and amplify conflicts between the two imperfect structural representations. I therefore selected a moderate operating point:

```python
controlnet_conditioning_scale = [0.4, 0.4]  # depth, FaceMesh
```

I checked this setting on additional inputs rather than treating one favorable example as sufficient.

![Multi-ControlNet example A](/images/newborn-generation/multi_1.png)
*Additional ultrasound example with depth/FaceMesh scales of 0.4/0.4.*

![Multi-ControlNet example B](/images/newborn-generation/multi_2.png)
*Another example using the same conditioning scales.*

**Takeaway:** 0.4/0.4 was a **practical choice for the tested examples**, not a universally optimal configuration. The pipeline still depends on the quality and agreement of the extracted conditions.

## 4. What did not work—and what it taught me

Not every technique that seemed promising improved the output. These are the decisions that most influenced the final direction.

| Attempt | Observed problem | Lesson / decision |
|---|---|---|
| **Img2Img** with moderate denoising | Pose drift from the ultrasound | Prefer text-to-image with explicit structural conditions |
| **Strong depth guidance** | Reinforced noisy or malformed eye geometry | Stronger conditioning cannot restore missing detail |
| **Full or strong FaceMesh guidance** | Over-constrained features; facial deformation | Prefer selective/local guidance and moderate scales |
| **Edge-only conditioning** | Some useful nose edges, less reliable head pose | Depth better served the global-pose role |
| **Strong dual-ControlNet guidance** | Competing constraints and new distortions | Tune the conditions jointly, not independently at maximum strength |

The overarching lesson was to **match each condition to the information it represents reliably**. Combining more conditions does not guarantee improvement; unreliable cues can compete.

## 5. Results and evaluation

### Qualitative results

The examples below illustrate the selected combined-conditioning setting. They are **not paired post-birth ground-truth comparisons** and should not be interpreted as identity reconstruction.

![Final qualitative example 1](/images/newborn-generation/multi_1.png)
*Example 1: output produced using the combined depth and FaceMesh conditions.*

![Final qualitative example 2](/images/newborn-generation/multi_2.png)
*Example 2: same conditioning-strength choice on another input.*

### Quantitative snapshot

For an exploratory assessment of the **final Multi-ControlNet + LoRA outputs**, I compared **41 generated images** with a reference set of **40 real newborn face crops** for appearance metrics. For landmark-based geometric metrics, **36 of the 41 ultrasound–output pairs** yielded usable landmark detections.

| Objective | Metric | Result | Interpretation |
|---|---|---:|---|
| Structural alignment | **Mean normalized landmark distance ↓** | **0.062** | Average distance relative to ultrasound face width, on detected pairs |
| Structural alignment | Inter-eye ratio relative error ↓ | 0.051 | Relative spacing error, estimated from landmarks |
| Head orientation | Pose geodesic difference ↓ | 12.06° | Estimated orientation gap, not anatomical ground truth |
| Appearance | CLIP image-to-image similarity ↑ | 0.893 | Semantic embedding similarity to reference newborn images |
| Landmark coverage | Detectable pairs | 36 / 41 (87.8%) | Five pairs omitted from landmark-based analysis |

Two additional appearance-distribution metrics were **FID = 176.32** and **KID = 0.1081 ± 0.0010**. With only ~40 images per set, both metrics need careful interpretation; they should **not** be treated as robust proof of photographic realism. Likewise, a high CLIP score is *not* a realism probability.

These scores help characterize the generated samples, but they **do not establish preservation of the baby's identity**. In particular, the structural comparison relies on landmark detectors operating across a large modality gap. There was **no matched quantitative baseline ablation** in this evaluation run, so the final-pipeline metrics alone cannot prove that the combined model outperforms every single-condition alternative.

<details>
<summary><strong>Additional structural metrics and evaluation notes</strong></summary>

| Landmark region | Normalized distance mean ± std ↓ |
|---|---:|
| Eyes | 0.067 ± 0.027 |
| Nose | 0.044 ± 0.022 |
| Mouth | 0.042 ± 0.028 |
| Face contour | 0.094 ± 0.036 |
| All reported keypoints | **0.062 ± 0.023** |

| Additional metric | Mean ± std |
|---|---:|
| Contour Chamfer distance ↓ | 0.110 ± 0.034 |
| Contour Hausdorff distance ↓ | 0.125 ± 0.040 |
| Nose Procrustes disparity ↓ | 0.022 ± 0.016 |
| Mouth Procrustes disparity ↓ | 0.016 ± 0.013 |
| Mean absolute pitch/yaw/roll gap ↓ | 5.80° ± 2.89° |
| Pose geodesic gap ↓ | 12.06° ± 5.84° |
| Aligned landmark-vector cosine similarity ↑ | 0.991 ± 0.006 |

Additional relative facial-ratio errors: mouth/face width 0.098; nose/face width 0.101; nose-to-mouth/face height 0.107; eye-to-nose/face height 0.203.

A human-scoring rubric was designed for realism, structure, artifact severity, and overall quality, but **human rating aggregates were not available** in this version. Grayscale pixel SSIM across ultrasound and photographs was not used as a primary metric because the modalities differ substantially.

</details>

## 6. Limitations and next directions

The biggest limitation is **information, not just model capacity**: ultrasound does not provide enough detail to determine how a newborn will actually look. The diffusion prior must invent texture and uncertain facial details.

Other practical limitations are **domain-mismatched depth/landmark estimators**, occasional failed or inaccurate landmark detection, artifacts from conflicting conditions, and **small-sample evaluation**. The structural metrics describe inferred geometry, not verified identity retention.

The most promising follow-ups would be:

1. **Ultrasound-adapted face or landmark extraction**, to reduce unreliable conditioning at the source.
2. **Region-specific control or masked conditioning**, especially around the eyelids, instead of global guidance everywhere.
3. **Systematic ablations on identical samples**, comparing baseline, depth only, FaceMesh only, combined guidance, and LoRA, supported by human ratings.
4. **Larger and better-curated data** to make evaluation more reliable; paired ultrasound/post-birth data would be especially valuable but difficult to obtain.

## 7. Key takeaways

- **Choose conditioning signals by their information content.** Depth captured coarse pose and nose geometry but did not reliably represent closed eyelids.
- **Investigate the representation before increasing model strength.** Comparing RGB-derived and ultrasound-derived depth helped reveal why the eyelid artifacts persisted.
- **Separate geometry from appearance.** FaceMesh/Depth addressed spatial structure; a newborn LoRA addressed apparent age and texture priors.
- **More guidance is not always better.** Moderate, complementary conditions gave a more useful trade-off than maximizing both ControlNet scales.

This project strengthened my experience in **computer-vision preprocessing, generative-model adaptation, experiment design, failure analysis, and quantitative evaluation**. More importantly, it reinforced a practical lesson: **the quality of a conditional generation system depends as much on the reliability of its inputs as on the generative model itself.**

---

**Tech stack:** Python · PyTorch · Diffusers · Stable Diffusion 1.5 · ControlNet · Depth Anything V2 · MediaPipe · OpenCV

*Research/demo note: Generated faces are plausible visualizations. They cannot predict or verify a baby's actual post-birth appearance.*