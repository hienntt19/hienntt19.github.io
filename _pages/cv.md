---
layout: single
title: "About"
permalink: /cv/
author_profile: true
---

## About Me

I am currently pursuing an **M.Sc. in Artificial Intelligence and Machine Learning at TU Darmstadt**, with a background in Computer Science and experience as an AI Engineer specializing in Computer Vision and Generative AI.

My experience spans practical ML development, from **dataset collection and annotation to model fine-tuning, evaluation, and deployment**. I have worked on object detection and diffusion-based image generation.

Beyond model development, I have built **end-to-end ML systems** involving asynchronous GPU inference, cloud infrastructure, CI/CD pipelines, and observability.

My primary interests are **Computer Vision, Generative AI, and ML Engineering**, particularly developing reliable AI solutions for real-world applications.

## Technical Interests

- **Computer Vision:** Object Detection
- **Generative AI:** Diffusion Models, Image Generation, LoRA, ControlNet
- **ML Engineering:** Model Serving, GPU Inference, MLOps, Cloud Infrastructure

## Technical Skills

- **ML Frameworks & Libraries:** PyTorch, OpenCV, NumPy, Pandas
- **Programming & Backend:** Python, SQL, FastAPI, Git, Linux
- **Cloud & MLOps:** GCP (GKE, Cloud Storage, Cloud SQL), Docker, Kubernetes, Helm, Terraform, Jenkins, RabbitMQ
- **Monitoring & Observability:** Prometheus, Grafana, ELK Stack, Jaeger


## Professional Experience

### AI Engineer | Vietnam-Korea Institute of Science and Technology (VKIST)
*Apr 2024 – Apr 2025*

Worked on Computer Vision and Generative AI applications, focusing on facial feature detection, custom dataset development, and diffusion-based image generation.

**Key Contributions:**
- Collected and curated a multi-angle Vietnamese portrait dataset covering diverse facial expressions, head poses, and lighting conditions.
- Trained and evaluated Faster R-CNN and YOLOv8 for facial feature detection, and developed an automated annotation pipeline using MediaPipe facial landmarks.
- Developed an ultrasound-conditioned newborn image generation pipeline using Stable Diffusion, LoRA, and ControlNet, investigating structural preservation and photorealism.

### Data Analyst Intern | UrBox
*Oct 2023 – Jan 2024*

Worked with analytical databases and SQL-based data processing to support business reporting and data quality.

**Key Contributions:**
- Extracted data from Amazon Redshift and ClickHouse using SQL to support requests across departments.
- Performed data validation, cleansing, and consistency checks to improve reporting accuracy and data reliability.


## Education

### M.Sc. Artificial Intelligence and Machine Learning
**Technical University of Darmstadt, Germany**  
*Apr 2026 – Present*

Areas of study: Deep Learning: Architecture and Method, Statistical Machine Learning, Multimodal AI.

### B.Sc. Computer Science (Honors)
**Vietnam National University – University of Engineering and Technology, Vietnam**  
*2019 – 2023*


## Selected Projects

### Realistic Newborn Photo Generation from Ultrasound Images

**Tech Stack:** Stable Diffusion 1.5, LoRA, Multi-ControlNet, Depth Anything V2, MediaPipe

Collected and curated newborn facial images, fine-tuned diffusion models, and developed a multi-condition generation pipeline integrating depth and facial landmarks. Conducted systematic experiments to assess image realism and structural consistency.

[Read Technical Deep Dive](https://hienntt19.github.io/posts/2025/01/newborn-generation/)

### Tsuki-Style Item Generation System

**Tech Stack:** Stable Diffusion, LoRA, FastAPI, GKE, RabbitMQ, Terraform, Jenkins, Prometheus, Grafana

Developed an end-to-end image generation system combining LoRA fine-tuning, asynchronous GPU inference, cloud infrastructure, automated deployment, and observability.

[View GitHub Repository](https://github.com/hienntt19/Game-item-generation-ML-system/tree/feature-2)

### Vietnamese Portrait Dataset & Facial Feature Detection

**Tech Stack:** YOLOv8, Faster R-CNN, MediaPipe, PyTorch

Developed a facial feature detection pipeline using a custom multi-angle Vietnamese portrait dataset and automated landmark-based annotation to support detection under challenging poses and lighting conditions.

### Virtual Try-On for Fashion Applications

**Tech Stack:** CATVTON, Stable Diffusion, PyTorch, FastAPI, Docker

Fine-tuned a diffusion-based virtual try-on model on a custom garment dataset and experimented with inference optimization techniques for efficient deployment.