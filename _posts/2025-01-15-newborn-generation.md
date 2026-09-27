---
title: "Generating Realistic Newborn Images from Ultrasound with Stable Diffusion 1.5"
date: 2025-01-15 16:00:00 +0200
categories: [Projects, Computer Vision]
tags: [stable-diffusion, controlnet, lora, computer-vision, generative-ai]
description: "Exploring how Stable Diffusion 1.5, ControlNet, Facial landmarks, and Depth conditioning can transform ultrasound images into realistic newborn portraits while preserving facial structure."
---

# Generating Realistic Newborn Images from Ultrasound with Stable Diffusion 1.5

> Can we generate a realistic newborn portrait from an ultrasound image while preserving its observable facial structure and head pose?

<!-- Add hero image / final pipeline result here -->
![Ultrasound to realistic newborn generation](/assets/img/newborn-generation/result.png)
_Example of the ultrasound-to-newborn generation result_

---

## 1. Introduction

Parents usually imagine and curious about how their kids would look like when seeing the ultrasound. To deal with this, the project aims at generating a realistic newborn portrait from a 3D/4D facial ultrasound image of a baby. 

The generation process goes under the constraints that the results can preserve the observable facial structure of the baby (facial geometry, eyes/mouth location, nose shape, maybe head pose) while can still create a plausible realistic and beautiful newborn image.

*Clarification: the target is not to predict the baby's exact post-birth appearance, it is relative appearance as the baby face changes quickly only after a few days, and ultrasound does not contain sufficient visual information for such reconstruction.

---

## 2. Problem Formulation

Input: A 3D/4D facial ultrasound image.

Output: A photorealistic newborn portrait.

Structural information to preserve:
  - Eyes and mouth position
  - Nose structure
  - Coarse facial proportions
  - Head pose (maybe)

Main challenge:
  - Domain gap: Ultrasound and real natural photographs belong to very different visual domains.
  - Incomplete information from ultrasound: Ultrasound images contain noise, missing texture/color information, ambiguous facial boundaries, weak or missing eyelid structure, etc.
  - Tradeoff between realism vs structural preservation: more ControlNet guidance gives better structure preservation but potentially more artifacts. Less guidance leads to better realism but weaker structural preservation. 

---

## 3. System Overview

Overall experimental pipeline:

<!-- IMAGE: clean architecture/pipeline diagram -->
![Ultrasound to realistic newborn generation](/assets/img/newborn-generation/pipeline.png)
_Pipeline of the ultrasound-to-newborn generation_

Responsibility of each component:
- Base diffusion model → photorealism
- Depth → pose and nose structure
- FaceMesh → eyes and mouth geometry
- LoRA → newborn appearance prior

---

## 4. Establishing a Baseline

At first, experiment with vanilla Stable Diffusion 1.5 baseline and Realistic Vision 5.1 baseline using the same prompt and configurations. I choose to compare those ones because Realistic Vision 5.1 is trained for generating realistic human images, therefore, it would be a promising baseline to try.

<div style="display: flex; gap: 20px; text-align: center;">

  <div style="flex: 1;">
    <img src="/assets/img/newborn-generation/sd.png"
         style="width: 100%;">
    <p><em>(a) Stable Diffusion 1.5 result</em></p>
  </div>

  <div style="flex: 1;">
    <img src="/assets/img/newborn-generation/rv.png"
         style="width: 100%;">
    <p><em>(b) Realistic Vision 5.1 result</em></p>
  </div>

</div>

Observations: 
- Vanilla Stable Diffusion 1.5 generates low light quality and unnatural baby face.
- Realistic Vision has stronger prior of human photorealism. The newborn appearance seems natural, skin texture is good and less artifacts.

So I chose Realistic Vision 5.1 for the pipeline's baseline model. Naturally, this baseline only can't produce structure-guided images, further experiment would be needed.

---

## 5. Structure-guided image generation
### Experiment 1 - Depth ControlNet

**Why depth and what depth estimator?**

Hypothesis: a depth representation can capture 3D facial geometry well. As a results, it would be helpful for keeping head pose, and possibly facial features.

Depth Anything V2 was selected as a more recent and robust monocular depth estimator, with stronger generalization capabilities than earlier MiDaS-based approaches. Since ControlNet Depth v1.1 is designed to support depth maps from different estimators rather than being tied to MiDaS, Depth Anything V2 can be used as the depth preprocessing stage.

Importantly, the goal here is not to recover metrically accurate 3D geometry from the ultrasound image. Instead, the depth map acts as a structural prior, providing relative spatial information about the visible facial surface-particularly the head pose, facial contour, nose, and mouth region-which can then be used to guide the Depth ControlNet during image generation.

Because ultrasound images are substantially different from the natural images used to train general-purpose monocular depth estimators, the predicted depth should therefore be interpreted as an approximate structural representation rather than true anatomical depth.

**Preprocessing pipeline**

![Depth processing pipeline](/assets/img/newborn-generation/depth_process.png)
_Depth processing pipeline_

The raw ultrasound image cannot be directly treated as a reliable depth condition. Ultrasound images contain substantial background regions and imaging noise, while the facial surface occupies only part of the image. I therefore constructed a preprocessing and post-processing pipeline to obtain a cleaner structural representation before using the depth map as a ControlNet condition.

The preprocessing pipeline contains the following steps:
1. Resize and padding: the original ultrasound image is resized while preserving its aspect ratio and padded to 512x512 instead of stretching to avoid geometric distortion of the fetal face.
2. Face-region masking: isolate the foreground (main facial region) from the dark background by using intensity thresholding, morphological closing and connected component analysis.
3. Denoising: reduce speckle-like noise from the ultrasound foreground using non-local means denoising. I also experimented with CLAHE for local contrast enhancement, but the ablation experiments showed little improvement in the resulting depth while sometimes emphasizing unwanted surface patterns. Therefore, CLAHE was excluded from the final pipeline.
4. Depth estimation: feed the preprocessed image into Depth Anything V2 (vitb) to obtain a relative depth map. Since the objective is to preserve facial structure rather than recover metric 3D geometry, the resulting depth map is treated as a structural representation of the visible facial surface.
5. Foreground normalization: normalize facial region only by calculating 1st-99th percentiles in face mask and clip to range [0, 1]. This prevents the ultrasound background and extreme depth values from dominating the available dynamic range, allowing variations around the nose, mouth, and facial surface to remain more visible.
6. Foreground separation: remap the facial region depth from [0, 1] -> [0.08, 1] while the background is explicitly set to 0. This small foreground floor prevents the farthest facial regions from collapsing to the same value as the background, preserving a distinguishable facial silhouette.
7. Boundary feathering: use a Gaussian blur mask and multiply it by the depth to soften the transition between the face and background, avoiding creating an overly sharp artificial edge.
8. Final depth map: the resulting map is converted to an 8-bit grayscale image and used as the depth condition for ControlNet. Rather than representing anatomically accurate depth, this processed map serves as a structural prior that provides the diffusion model with information about the overall head pose, facial contour, and coarse geometry of the visible facial features.

**Experiments**

At first, I experiment with different controlnet scale to determine the best one. The following image shows the results of different controlnet scale (other configurations remain the same as baseline experiment).

![Depth scale](/assets/img/newborn-generation/depth_scale.png)
_Results of different controlnet scale_

Observations:

- The results show that even at a relatively low conditioning scale of 0.3, depth controlnet begins to influence the generated head pose and facial structure. At around 0.5, the generated face follows the depth condition more clearly while still maintaining reasonable visual quality.

- However, two limitations become apparent:

  - The eye region is not reliably preserved. Some generated images contain distorted eyes, duplicated eyelid lines, or eyes appear partially open.

  - The generated face does not always exhibit newborn-specific characteristics. Although the base model produces realistic baby faces, some outputs appear closer to older infants than newborns.

These two limitations originate from different aspects of the pipeline and therefore require different solutions.

**Investigating the eye region: RGB vs ultrasound depth**

Before introducing additional conditioning, I investigated whether the eyelid artifacts originate from the depth representation itself.

As a qualitative comparison, I estimated a depth map from a real RGB image of a sleeping newborn, where the closed eyelids are clearly visible in the original image.

![RGB Depth mao](/assets/img/newborn-generation/rgb_depth.png)
_RGB image depth map_

Interestingly, although the eyelids are clearly visible in the RGB image, their boundaries are almost absent from the estimated depth map. In contrast, larger 3D structures such as the head shape, nose, cheeks, and facial contour remain clearly represented.

This suggests that monocular depth estimation is effective at capturing coarse 3D facial geometry, but does not necessarily preserve fine appearance boundaries such as closed eyelids. The geometric depth variation around a closed eyelid is very small compared with structures such as the nose or overall facial surface.

I then compared this behavior with depth maps estimated from 3D ultrasound images.

![Ultrasound Depth mao](/assets/img/newborn-generation/us_depth.png)
_Ultrasound image depth map_

The problem becomes more challenging in the ultrasound domain. In addition to the inherently weak depth variation around the eyelids, ultrasound images often contain ambiguous boundaries, missing surface information, shadows, and imaging artifacts. As a result, the estimated depth map may either lose the eyelid structure entirely or encode surrounding artifacts as unreliable geometric cues.

This comparison indicates that the eye-region problem is not simply caused by Depth ControlNet itself. Instead, there is an information limitation in the conditioning representation: fine eyelid geometry is weak even when depth is estimated from a clean RGB image, and becomes even less reliable when estimated from ultrasound.

**Failure case 1: unreliable eyelid geometry -> Facemesh guidance**

Although the depth map captures coarse structures such as the head pose, nose, mouth and facial contour, the eye region is considerably less reliable.

Because fine eyelid geometry is weak or absent in the depth representation, the final appearance of the eyes is largely influenced by the generative prior of the base model. When the depth condition around the eye region is additionally noisy or ambiguous, as can occur with ultrasound-derived depth, this interaction may produce unstable structures such as:

- duplicated or multiple eyelid lines,
- distorted eye shapes,
- partially open-looking eyes.

Increasing the ControlNet conditioning strength cannot recover geometric information that is already missing from the depth representation. Instead, stronger conditioning may reinforce unreliable structures encoded in the estimated depth map.

Therefore, depth conditioning alone is insufficient for controlling fine eye geometry.

Proposed Solution: FaceMesh ControlNet

To compensate for the unreliable eye information in the depth map, I introduce an additional FaceMesh-based condition that explicitly provides the approximate location and shape of the closed eyes.

The two ControlNet conditions therefore serve complementary roles:

- Depth ControlNet: head pose and coarse 3D facial geometry.
- FaceMesh ControlNet: explicit guidance for fine eye geometry.

This allows the depth representation to focus on the structures it captures reliably, while FaceMesh provides additional spatial information for the eye region.


**Failure case 2: incorrect age/appearance -> newborn LoRA**

Solving the structural guidance problem does not address another limitation observed in the generated images: the baby's apparent age and sleeping-newborn appearance.

Depth and FaceMesh primarily provide geometric constraints. They describe where facial structures should be located, but they do not explicitly determine whether the generated face should resemble a newborn or an older infant.

These semantic and appearance-related characteristics remain largely dependent on the learned prior of the base model. Although Realistic Vision can generate photorealistic baby faces, it does not consistently produce the specific appearance targeted in this project: a young newborn with naturally closed eyes.

Proposed Solution: Newborn LoRA

To introduce a stronger newborn-specific appearance prior, I fine-tune a LoRA on images of sleeping newborns.

The LoRA is intended to improve:

- newborn-specific facial characteristics,
- age-appropriate appearance,
- the visual appearance of naturally closed eyes during sleep,

while structural preservation remains primarily handled by the ControlNet conditions.

Therefore, FaceMesh and the newborn LoRA address two different aspects of the eye problem:

- FaceMesh provides geometric guidance for where and how the closed eyes should be positioned, while the newborn LoRA provides an appearance prior for what naturally closed eyes on a sleeping newborn should look like.

---

### Experiment 2 - FaceMesh ControlNet

**Preprocessing pipeline** 

![FaceMesh processing pipeline](/assets/img/newborn-generation/fm_process.png)
_FaceMesh processing pipeline_

To provide additional guidance for fine facial structures, particularly around the eyes and mouth, a FaceMesh condition is extracted using MediaPipe Face Landmarker.

The preprocessing pipeline consists of the following steps:

1. Resize and padding: resize the ultrasound image while preserving its aspect ratio and pad it to 512 × 512. The same transformation as the depth preprocessing pipeline is used to ensure spatial alignment between the depth and FaceMesh conditions.

2. Facial landmark detection: apply MediaPipe Face Landmarker to detect facial landmarks. Lower detection and face-presence confidence thresholds (0.1) are used because fetal faces in ultrasound images are more difficult to detect than faces in natural images.

3. CLAHE fallback: if landmark detection fails, apply CLAHE to enhance local contrast and perform detection again.

4. FaceMesh construction: convert the detected landmarks into a FaceMesh condition consisting of facial tessellation and contours for the face oval, eyes, eyebrows, and lips. Iris connections are excluded because the fetal eyes are closed and iris information is not reliably observable in the ultrasound images.

4. Alignment verification: overlay the generated FaceMesh on the ultrasound image to visually verify that the detected landmarks are aligned with the observable facial structures.

The resulting FaceMesh image is then used as an additional structural condition for the FaceMesh ControlNet.

**Experiments**

Similar to the Depth ControlNet experiment, I tested different FaceMesh conditioning scales to determine a suitable level of structural guidance.

![FaceMesh results with different conditioning scale](/assets/img/newborn-generation/fm_scale.png)
_FaceMesh results with different conditioning scale_

Observations:

- At a low conditioning scale (0.3), the generated face begins to follow the FaceMesh condition, particularly in head orientation and the relative positions of the eyes and mouth, while still retaining considerable freedom from the base model.

- At moderate scales (0.5–0.7), the generated facial structure follows the FaceMesh more clearly. The closed-eye geometry and mouth position become more consistent with the extracted landmarks while the images remain visually realistic.

- At a high conditioning scale (1.0), the model follows the FaceMesh too strongly. This introduces visible distortions around the eye and mouth regions and reduces the natural appearance of the generated face.

- Overall, 0.5–0.7 provides a reasonable trade-off between structural guidance and visual quality. A scale of 0.5 was selected for subsequent experiments to avoid over-constraining the generation.


**Why use FaceMesh as a local constraint?**

The experiment suggests that FaceMesh is most useful for providing explicit geometric anchors for facial features such as the eyes and mouth. These structures are difficult to recover reliably from the estimated depth map, while their positions and shapes are represented more explicitly by facial landmarks.
H
owever, constraining the entire facial mesh too strongly can reduce the diffusion model's freedom to generate natural facial details and may introduce distortions. In addition, the FaceMesh condition provides limited information about the 3D surface geometry of the nose compared with the depth condition.

FaceMesh is more useful as a local geometric constraint than as a complete representation of the facial structure. Therefore, I use FaceMesh primarily to guide fine facial features, while relying on depth information for the overall head pose and surface geometry.

---

### Experiment 3 - Adding Newborn LoRA

**Motivation**

Although Realistic Vision V5.1 already generates realistic baby portraits, some outputs resemble older infants rather than newborns. Therefore, a LoRA was fine-tuned to strengthen the newborn appearance prior.

The components of the pipeline have different roles:

- Depth ControlNet: head pose and coarse facial geometry
- FaceMesh ControlNet: local facial features, particularly the eyes and mouth
- Newborn LoRA: newborn-specific appearance

The LoRA is therefore used as an appearance prior rather than a structural constraint.

**Dataset**

The LoRA was fine-tuned on approximately 40 face-focused images of Asian newborns. Cropping reduces irrelevant background information and focuses the adaptation on newborn facial characteristics.

Each image was paired with an image-specific caption generated with the assistance of an LLM.

**Training**

The LoRA was trained on the same Realistic Vision V5.1 base model used in the final pipeline.

| Parameter | Value |
|---|---|
| Training images | ~40 |
| Resolution | 512 × 512 |
| LoRA rank | 8 |
| Batch size | 2 |
| Gradient accumulation | 2 |
| Effective batch size | 4 |
| Training steps | 600 |
| Learning rate | 5 × 10⁻⁵ |
| LR scheduler | Constant with warmup |
| Warmup steps | 30 |
| Precision | FP16 |
| Augmentation | Random horizontal flip |
| Seed | 42 |

A relatively small rank of 8 and conservative learning rate were used because the goal was a lightweight domain adaptation rather than substantially modifying the base model.

**LoRA Scale Experiments**

The trained LoRA was tested at different strengths while keeping the seed and other generation parameters fixed.

![LoRA under different scales](/assets/img/newborn-generation/lora_scale.png)
_Effect of increasing LoRA strength on the generated newborn appearance._

Increasing the LoRA strength produces a clear progression toward the training distribution:

- 0.0: realistic baseline, but the face appears closer to an older infant.
- 0.3: slightly rounder face, fuller cheeks, and softer features.
- 0.5: stronger newborn characteristics while retaining realistic facial detail.
- 0.7: further strengthens the newborn appearance, but the improvement over 0.5 becomes smaller.
- 1.0: the LoRA becomes dominant, producing a smoother and rounder face while reducing some facial variation.

Overall, the LoRA successfully shifts the base model toward rounder facial proportions, fuller cheeks, and softer newborn-like features. However, stronger LoRA conditioning is not necessarily better. Since the dataset contains only about 40 images, excessive LoRA influence may reduce diversity and overemphasize characteristics of the training set.

A scale of 0.5 was therefore selected for subsequent experiments as a balance between strengthening the newborn appearance and preserving the photorealistic prior of Realistic Vision.

Importantly, the LoRA does not preserve the geometry of the ultrasound face. Its role is complementary to ControlNet:

> ControlNet guides how the face is structured, while LoRA biases what the newborn looks like.

---

### Experiment 4 - Multi-ControlNet

The previous experiments show that Depth and FaceMesh provide complementary structural information. Therefore, both conditions are combined using a Multi-ControlNet pipeline.

Their roles are defined as follows:

- Depth ControlNet
  - Preserve the overall head pose.
  - Provide coarse facial geometry.
  - Guide the 3D surface structure, particularly around the nose.

- FaceMesh ControlNet
  - Provide explicit geometric guidance for the eyes.
  - Guide the position and shape of the mouth.

The contribution of each ControlNet is controlled using:

```python
controlnet_conditioning_scale=[depth_scale, facemesh_scale]
control_guidance_start=[0.0, 0.0]
control_guidance_end=[depth_end, facemesh_end]
```

`controlnet_conditioning_scale` determines the strength of each structural condition. A higher value forces the generation to follow the corresponding condition more strongly, while a lower value gives the diffusion model more freedom.

For the conditioning-scale experiment, the remaining generation parameters were kept fixed so that the interaction between Depth and FaceMesh could be studied independently.

**Conditioning Scale Experiment**

Different combinations of Depth and FaceMesh conditioning scales were tested.

![Depth vs FaceMesh conditioning scale](/assets/img/newborn-generation/d_fm_scale.png)
_Depth and FaceMesh combinations under different conditioning scales._

The results show several trends:

- Increasing the Depth scale strengthens control over the overall head pose and coarse facial geometry, particularly around the nose and central facial structure.
- Increasing the FaceMesh scale introduces stronger constraints around landmark-defined regions such as the eyes and mouth.
- Increasing both scales simultaneously does not necessarily improve the result. Strong combined conditioning can over-constrain the generation and introduce artifacts, particularly around the eyelids.
- Moderate conditioning provides a better balance between structural guidance and the generative freedom of the base model.

This reveals an important trade-off:

> Structural fidelity ↔ Generative freedom

Rather than maximizing both conditioning strengths, the objective is to use sufficient guidance to preserve observable facial structure while still allowing the diffusion model to generate natural facial details.

**Selected Configuration**

Based on the scale sweep, the following configuration was selected for the final pipeline:

```python
controlnet_conditioning_scale=[0.4, 0.4]
```

To check whether this configuration generalizes beyond a single example, the same setting was applied to additional ultrasound samples.

![Multi-ControlNet selected scale example 1](/assets/img/newborn-generation/multi_1.png)
_Depth = 0.4 and FaceMesh = 0.4 on the first example._

![Multi-ControlNet selected scale example 2](/assets/img/newborn-generation/multi_2.png)
_Depth = 0.4 and FaceMesh = 0.4 on another ultrasound example._

Across these examples, the combined model provides a useful compromise between the two conditioning signals. Depth contributes global pose and coarse surface geometry, while FaceMesh provides more explicit guidance for local facial features.

The 0.4 / 0.4 configuration is therefore used as a practical operating point rather than a universally optimal setting. It provides useful structural guidance from both conditions without excessively constraining the diffusion model.

**Mini Conclusion**

Depth and FaceMesh address different weaknesses of the conditioning pipeline:

```
Depth     → global pose + coarse facial geometry
FaceMesh  → local eye + mouth geometry
                         ↓
                  Multi-ControlNet
                         ↓
          balanced structural guidance
```

Combining them provides more complete structural guidance than either condition alone, while moderate conditioning strengths help avoid artifacts caused by over-constraining the generation.

---

## 6. Final Pipeline

```text
Ultrasound
    ↓
Preprocessing
    ↓
Depth ───────────────┐
                     │
Eyes + Mouth FaceMesh├── Multi-ControlNet
                     │
                     ↓
             Realistic Vision V5.1
                     +
               Newborn LoRA
                     ↓
             Final Generation
```

Document final hyperparameters:

```text
Base model:
Scheduler:
Inference steps:
Guidance scale:

Depth scale:
Depth guidance end:

FaceMesh scale:
FaceMesh guidance end:

LoRA scale:

Seed strategy:
```

<!-- IMAGE: final pipeline + several generated examples -->

---

## 7. Evaluation

I separate evaluation into two complementary objectives: **image realism** (does the output look like a photoreal newborn?) and **structural preservation** (does the output retain the observable facial geometry of the input ultrasound?). All quantitative results below are reported for our final system, **Multi-ControlNet + LoRA**, on a held-out ultrasound set with paired generated images (*N* = 41 generated; 40 real newborn crops for distributional metrics).

### 7.1 Image Realism

Realism is assessed against a reference set of real Asian newborn face crops using distributional metrics, CLIP-based semantic similarity, and a human rating protocol.

**FID / KID.**  
We compute Fréchet Inception Distance (FID) and Kernel Inception Distance (KID) between generated images and real newborn crops (Inception features; KID subset size = 40). Lower is better for both.

| Metric | Value |
|---|---|
| FID ↓ | 176.32 |
| KID mean ↓ | 0.1081 ± 0.0010 |
| # real / # generated | 40 / 41 |

With a modest sample size, FID is high-variance and should be interpreted cautiously; KID is therefore the more informative distributional score in this setting. The numbers indicate that generated faces occupy a related but still distinguishable region of the newborn appearance manifold—consistent with a method that must invent photoreal texture while being constrained by ultrasound geometry.

**CLIP-based similarity to newborn appearance.**  
Using OpenCLIP ViT-B/32, each generated image is scored by (i) cosine similarity to an ensemble of newborn-appearance text prompts, (ii) a margin against negative prompts (ultrasound / blur / adult), and (iii) cosine similarity to the mean embedding of the real newborn set.

| CLIP score | Mean | Std |
|---|---|---|
| Text similarity ↑ | 0.333 | 0.007 |
| Text margin (pos − neg) ↑ | 0.085 | — |
| Image–image similarity to real newborns ↑ | **0.893** | 0.020 |

The strong image–image CLIP similarity (**0.89**) suggests that generated faces are semantically close to real newborn portraits in embedding space, while the positive text margin confirms preference for “newborn photo” over “ultrasound / adult” descriptions.

**Human visual assessment.**  
We use a 1–5 Likert protocol (template released with the evaluation artifacts):

| Axis | Rubric |
|---|---|
| Realism | 1 = synthetic / ultrasound-like → 5 = photoreal newborn |
| Identity / structure | 1 = structure lost → 5 = structure clearly preserved |
| Artifact-free | 1 = heavy artifacts → 5 = clean |
| Overall | 1 = poor → 5 = excellent |

Raters score each generated image independently. We recommend ≥2–3 raters and reporting mean ± std. *(Quantitative human means are filled after annotation of `7_1_human_rating_filled.csv`.)*

### 7.2 Structural Preservation

Structural fidelity is measured by pairing each generated image with its source ultrasound (matched by filename stem), detecting MediaPipe FaceMesh landmarks on both (CLAHE fallback for low-contrast ultrasound), and comparing geometry after face-width normalization. Of 41 generated images, landmarks were successfully recovered on **36** pairs (**87.8%** detection rate; 5 failures on either ultrasound or generated face).

**Facial landmark distance / normalized landmark error (NLD).**  
Mean Euclidean landmark error normalized by ultrasound face width (↓ better):

| Region | NLD mean | Std |
|---|---|---|
| Eyes | 0.067 | 0.027 |
| Nose | 0.044 | 0.022 |
| Mouth | 0.042 | 0.028 |
| Face contour | 0.094 | 0.036 |
| **Mean keypoints** | **0.062** | 0.023 |

Nose and mouth stay within ~4% of face width on average; the outer contour is harder (ultrasound silhouette vs. photo hairline/cheeks), which pulls the global mean to **0.062**.

Additional shape metrics on contours:

| Metric | Mean | Std |
|---|---|---|
| Contour Chamfer (norm.) ↓ | 0.110 | 0.034 |
| Contour Hausdorff (norm.) ↓ | 0.125 | 0.040 |
| Nose Procrustes disparity ↓ | 0.022 | 0.016 |
| Mouth Procrustes disparity ↓ | 0.016 | 0.013 |

**Eye / mouth relative positions.**  
We compare classical facial ratios (relative error, ↓ better):

| Ratio | Rel. error mean |
|---|---|
| Inter-eye / face width | **0.051** |
| Mouth width / face width | 0.098 |
| Nose width / face width | 0.101 |
| Nose→mouth / face height | 0.107 |
| Eye→nose / face height | 0.203 |

Horizontal spacing (especially inter-eye distance) is preserved well (~5% relative error). Vertical mid-face proportions (eye→nose) are noisier—expected when ultrasound foreshortening and soft-tissue visibility differ from photographic portraits.

**Head-pose difference.**  
From MediaPipe facial transformation matrices we report absolute Euler gaps and the geodesic rotation angle between ultrasound and generated poses:

| Pose metric | Mean | Std |
|---|---|---|
| Mean \|Δpitch, Δyaw, Δroll\| (°) ↓ | 5.80 | 2.89 |
| **Geodesic rotation (°) ↓** | **12.06** | 5.84 |

A ~12° mean geodesic gap indicates that overall head orientation is largely retained, with residual drift typical of generative pose freedom under moderate ControlNet scales.

**Structural similarity on extracted representations.**  
After Procrustes alignment of a joint landmark set (oval + eyes + nose + lips), we measure cosine similarity of the aligned coordinate vectors:

| Representation metric | Mean | Std |
|---|---|---|
| **Cosine similarity ↑** | **0.991** | 0.006 |
| MSE (aligned) ↓ | 9.1×10⁻⁵ | 5.6×10⁻⁵ |
| Procrustes disparity ↓ | 0.019 | 0.011 |

Near-unit cosine similarity shows that the *shape graph* of the face is tightly preserved even when absolute pixel placement (NLD) still reflects framing and domain shift. Pixel-domain grayscale SSIM between ultrasound and photo is near zero by construction (different modalities) and is **not** used as a primary score.

### 7.3 Results Summary (Multi-ControlNet + LoRA)

<!-- TABLE: quantitative evaluation -->

| Objective | Metric | Score |
|---|---|---|
| Realism | FID ↓ | 176.32 |
| Realism | KID ↓ | 0.108 ± 0.001 |
| Realism | CLIP text sim ↑ | 0.333 |
| Realism | CLIP ↔ real newborn ↑ | **0.893** |
| Structure | Face detection rate ↑ | 87.8% (36/41) |
| Structure | Mean keypoint NLD ↓ | **0.062** |
| Structure | Inter-eye ratio relerr ↓ | 0.051 |
| Structure | Pose geodesic ↓ | 12.1° |
| Structure | Landmark repr. cosine ↑ | **0.991** |
| Human | Likert (realism / structure / overall) | *pending annotation* |

**Takeaway.**  
Multi-ControlNet + LoRA produces images that are **semantically newborn-like** (CLIP image similarity ≈ 0.89) while **keeping ultrasound facial geometry**: average normalized landmark error ≈ 6% of face width, inter-eye proportions within ~5%, and Procrustes-aligned landmark cosine ≈ 0.99. Remaining gaps are mainly photoreal distributional distance (FID/KID under small *N*) and occasional landmark failures on difficult ultrasound frames—both natural limits of the ultrasound→photo domain gap rather than collapse of the conditioning signal.

**Limitations.**  
(1) FID/KID with ~40 images per set are noisy; we emphasize KID and CLIP.  
(2) Landmark metrics require successful detection on *both* domains (5/41 pairs dropped).  
(3) Human scores are required for perceptual claims beyond automatic metrics.  
(4) No ablation table is reported here because only the final Multi-ControlNet + LoRA outputs were evaluated in this run.


---

## 8. What Did Not Work

This section is important for the portfolio.

### Img2Img

- tested with moderate denoising strength
- generated images drifted from the ultrasound pose
- therefore returned to text-to-image + explicit structural conditioning

### Strong Depth Conditioning

- preserved structure better
- amplified malformed facial geometry

### Full FaceMesh

- over-constrained the face
- reduced realism
- caused facial distortions

### Edge-only Conditioning

- useful nose information
- poor pose preservation

### Strong Multi-ControlNet

- competing structural constraints
- increased deformation

### Lesson

Explain that conditional diffusion requires balancing:

> enough conditioning to preserve structure, but enough freedom for the diffusion prior to generate a natural face.


---

## 9. Results

Show the strongest qualitative examples.

### Example 1

```text
Ultrasound → Depth → FaceMesh → Generated
```

### Example 2

```text
Ultrasound → Depth → FaceMesh → Generated
```

### Example 3

```text
Ultrasound → Depth → FaceMesh → Generated
```

Also include difficult/failure cases rather than only cherry-picked successes.

---

## 10. Key Findings

Summarize the technical findings:

1. A strong diffusion model can generate realistic newborn images without fine-tuning.
2. Text conditioning alone does not preserve ultrasound facial structure.
3. Depth conditioning helps preserve global pose but can introduce fine-detail artifacts.
4. Sparse FaceMesh conditioning works well for eyes and mouth.
5. Full facial conditioning can over-constrain the diffusion process.
6. Combining multiple ControlNets requires careful balancing.
7. Stronger conditioning does not necessarily mean better structural preservation.
8. LoRA is most useful as an appearance prior rather than a structural constraint.

---

## 11. Limitations

Be explicit about what the system cannot claim.

- Ultrasound does not contain enough visual information to reconstruct an exact future newborn appearance.
- Some facial details are inherently ambiguous.
- Depth estimation models are not designed specifically for ultrasound.
- Facial landmark detectors may fail or introduce inaccurate geometry.
- Generated texture and fine facial appearance are influenced by the diffusion prior.
- Evaluation of identity preservation is difficult because paired ultrasound/newborn datasets are limited.

---

## 12. Future Work

Possible directions:

- ultrasound-specific depth estimation
- ultrasound-specific facial landmark detection
- region-specific conditioning
- ControlNet / adapter trained directly on ultrasound structure
- attention or mask-based regional control
- larger paired ultrasound/newborn datasets
- improved quantitative structural metrics
- SDXL / newer diffusion architectures
- identity/geometry-aware conditioning

---

## 13. Conclusion

Summarize the project around one central idea:

> Generating a realistic newborn portrait from ultrasound is not simply a photorealistic image-generation problem. The key challenge is finding the right balance between structural conditioning from an inherently ambiguous ultrasound image and the strong visual prior provided by a diffusion model.

Briefly summarize what was learned from Depth, FaceMesh, Multi-ControlNet, and LoRA.

---

## Code & Resources

**GitHub:** [Project Repository](#)

**Tech Stack:** Python · PyTorch · Diffusers · Stable Diffusion · ControlNet · MediaPipe · OpenCV

---

*This project explores generative modeling and structural conditioning from ultrasound imagery. Generated images represent plausible visualizations rather than predictions of a baby's exact physical appearance.*