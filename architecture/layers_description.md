## Layers

### Presentation Layer

A modular architectural approach was adopted to ensure maintainability and extensibility. Separating core functions into distinct components allows for more efficient development and testing.

The system follows the Model-View-ViewModel (MVVM) presentation pattern, which separates the user interface (View) from the underlying logic (Model) via an intermediary (ViewModel). This separation of concerns promotes modularity and enables independent development, testing, and maintenance of each component (KOURAKLIS, 2016). MVVM has been successfully applied in medical software applications (SHEIKH and SHEIKH, 2019; TRECEÑO-FERNÁNDEZ et al., 2021; YUE et al., 2024), providing a structured framework for managing complex data flows and user interactions.

The **View** handles the graphical interface, user interactions, and representation states. It defines the structure and layout while delegating logic and data handling to the ViewModel. Through data binding, the interface stays synchronized with the application state without direct coupling.

The **ViewModel** mediates between the View and Model. It manages application state, processes user input, and exposes commands and observable properties for data binding. It adapts data from the Model for display and relays user actions back to the Model.

The **Model** contains the core functionality of the tool, including image-processing routines, metric calculations, and data management. It also manages orchestration logic and the plugin framework that define internal operations. By keeping the Model independent from presentation logic, computational processes remain unaffected by user-interface changes. In this architecture, the Model spans the deeper system layers.

### Application Layer

The Application or Orchestration Layer coordinates study processing. It interprets user commands or external triggers, schedules jobs, validates and routes parameters, manages retries and timeouts, and invokes domain services. It also manages plugin lifecycles, including discovery, versioning, and activation; enforces execution policies, such as CPU versus GPU processing and batch versus single-series processing; and records audit events.

This layer does not implement image analysis itself. Instead, it delegates analysis to the Domain Layer, aggregates results, and forwards them to the Integration and Presentation Layers.

The **Plugin Manager** governs plugin orchestration logic. It decides which plugins to use for a given workflow, manages their execution sequence, and integrates their outputs into the broader processing pipeline.

### Domain Layer

The Domain Layer, also referred to as the Business Layer, contains the core image-analysis logic and domain rules. These include concrete implementations of segmentation, model execution, tissue characterization, automatic region-of-interest (ROI) selection, MTF/NPS/PS computations, and quality-labeling logic. It exposes well-defined interfaces that are called by the Application Layer.

The Domain Layer should remain free of persistence and network concerns. It receives inputs and returns outputs, while storage, logging, and external communication are handled by the Infrastructure and Integration Layers.

### Integration Layer

The Integration Layer connects the application to external clinical systems. It implements adapters for DICOM, HL7, and API-based communication, enabling integration with hospital infrastructures.

Its functions include:

- Retrieving CT studies from PACS.
- Writing results back into DICOM objects through private tags or Structured Reports. Write Excel/ASCII reports in a dedicated folder on hospital network.
- Sending reports to hospital information systems (HIS) via HL7.
- Publishing quality reports and notifications to external systems. - still in consideration

Note: this DICOM/HL7-based communication with PACS/HIS is distinct from the internal plugin/hook API described under FR-08 and the Domain Layer's Core API. The Integration Layer's adapters are protocol clients that exchange data with external clinical systems; the plugin/hook API is an in-process Python interface used to extend BICTool's own analysis modules and is not network-exposed.

### Infrastructure Layer

The Infrastructure Layer provides shared technical services for all other layers. It includes file and database management, hardware-resource allocation, GPU scheduling, multiprocessing, configuration handling, and centralized logging and auditing.

It enforces de-identification and secure-caching policies to protect protected health information (PHI), even in offline environments. Higher layers access these services through defined interfaces, ensuring loose coupling and maintainability.

The **Plugin Loader** is responsible for the technical aspects of plugin handling. It dynamically discovers available modules, validates their interfaces, and loads them into isolated environments to ensure stability and security.

## Tools and Libraries

This section describes the preliminary set of tools and libraries proposed for the development of the image-quality-assessment tool. The tools selected at this stage are subject to change based on further evaluation and testing during the development process.

### Programming Environment

To ensure cross-platform compatibility across Windows, Linux, and macOS, and to leverage the extensive ecosystem of scientific and artificial-intelligence libraries, Python was selected as the primary development language. Python is widely used in the medical-imaging community for image processing (WIDODO, ADI and GERNOWO, 2020), machine learning (PÉREZ-GARCÍA, SPARKS and OURSELIN, 2021), and data analysis (PÉREZ, GRANGER and HUNTER, 2011).

Python’s integration capabilities with AI frameworks, such as PyTorch and TensorFlow, and medical-imaging toolkits, such as MONAI, SimpleITK, and pydicom, make it suitable for extending the software with advanced analysis modules, including segmentation and tissue characterization.

Python is generally slower than compiled languages because it is interpreted and dynamically typed, which adds computational overhead during execution. For operations where performance is critical, selected parts of the code can be written in C++ and connected to Python through bindings. This allows those sections to run almost as efficiently as native C++ code.

### User Interface Framework

**Decision (2026-09-10): PySide6.** The user interface is implemented with PySide6 while following the MVVM architectural pattern. This separation of logic ensures that visualization, user interaction, and application control remain decoupled, thereby facilitating modular extension and testing.

PySide is licensed under the permissive GNU Lesser General Public License (LGPL), allowing its use in both open-source and proprietary applications without requiring a commercial license — unlike PyQt, which is GPL/commercial dual-licensed. This matters for FR-08's in-process plugin API: LGPL PySide does not force third-party plugin authors (e.g. a hospital's in-house extension) into GPL, whereas linking against GPL PyQt would.

PySide6 (Qt6) is the actively maintained line and was chosen over PySide2 (Qt5, upstream-EOL) to prioritize long-term security/maintenance for software with a multi-year clinical deployment horizon. The tradeoff, accepted knowingly: **Qt6 requires Windows 10 (1809+) or newer and cannot run on Windows 7/8**, which some hospital workstations may still use (NFR-04 requires Windows/macOS/Linux compatibility but does not pin a minimum OS version). If Windows 7/8 support turns out to be a hard requirement for a deployment site, the fallback is a downgrade to PySide2 (Qt5) — both are LGPL, so the licensing rationale above holds for either; only the OS-floor/maintenance tradeoff changes. This should be revisited if/when a concrete hospital deployment target contradicts the Windows 10+ assumption.

For volumetric visualization and image inspection, the system may integrate Napari and SimpleITK. Napari is an interactive viewer that provides high-performance, GPU-accelerated 2D and 3D rendering. It also supports custom plugins for visualizing segmentation masks and metric overlays.

SimpleITK provides reliable tools for reading, writing, and processing medical images, including resampling and filtering. It connects raw DICOM data to analysis workflows, allowing medical images to be more easily integrated into quantitative and algorithmic pipelines.

### Image Processing and Analysis

The analytical modules, including segmentation, ROI identification, and metric computation, are implemented using common scientific libraries:

- **NumPy** and **SciPy** for array operations and numerical routines.
- **scikit-image** for image-processing utilities.
- **PyTorch** for model-based or convolutional-neural-network (CNN)-driven segmentation tasks.

These components are wrapped within the Domain Layer and exposed through a uniform plugin interface. This allows new analysis methods to be integrated without modifying the application core.

Segmentation is the first and most important step in image analysis because it provides precise isolation of the structures required for quantitative assessment. In MRI, segmentation is relatively straightforward because of the higher soft-tissue contrast. Several semi-supervised and auxiliary-data methods have been developed, such as the probabilistic refinement with pseudo-labels proposed by ITO et al. (2019) and the SLANT tiled network architecture implemented by HUO et al. (2019) for efficiently processing high-resolution 3D volumes.

In contrast, CT segmentation presents greater challenges, including lower soft-tissue contrast, skull artifacts, and anatomical variability. To address these issues, hybrid and unsupervised methods have been proposed. HU et al. (2024) developed a model that combines traditional image processing with CNN-based classification for rapid and accurate segmentation. SANTOS et al. (2024), building on KIM, KANEZAKI and TANAKA (2020), optimized an unsupervised framework that segments images without requiring manual labels.

Although an optimal CT-segmentation approach remains an open research question, a CNN-based or hybrid method will be implemented in this project.

**Decision (2026-09-16): TensorFlow/Keras, not PyTorch, for the CNN segmentation modules.** This section's PyTorch mention above was preliminary (see this section's own opening caveat, "subject to change based on further evaluation"). The CNN segmentation work that actually got implemented wires in two existing pretrained models - `jasonccai/HeadCTSegmentation` (17-class head-CT anatomy segmentation, fulfilling FR-02) and a transfer-learned stroke-lesion variant built on the same encoder - both published and distributed as TensorFlow/Keras models with real trained weights and no PyTorch port available. Re-deriving or re-porting real trained weights into a different framework is itself a risky, effortful, error-prone undertaking with no real benefit for reusing an already-validated model, so TensorFlow/Keras was used as delivered rather than forcing a framework match. See `source/bictool/domain/segmentation/_unet_architecture.py`'s module docstring for the architecture-reproduction details. PyTorch remains the reasonable default assumption for any future segmentation work that involves training a genuinely new model from scratch rather than wiring in an existing one.

The metrics currently proposed in WP1 and described in Chapter 2 can only be applied partially to the relevant anatomical structures:

- The falx and septum pellucidum are suitable for MTF and PS analysis.
- The ventricles and white matter can be used for NPS analysis.
- For the basal ganglia, the first step is to determine whether these structures are visible in the images.
- One initial approach is to use a brain atlas on the pre-segmented region to locate the basal ganglia.
- Once their visibility has been confirmed, contrast can be assessed using intensity profiles or statistical measures.
- Noise can be evaluated in nearby homogeneous areas.

### Communication

Compatibility with clinical workflows is ensured through standard medical-communication libraries:

- **pydicom** is used for reading, writing, and modifying DICOM files, including the insertion of computed metrics into private DICOM tags or Structured Reports.
- **HL7apy** provides parsing and serialization of HL7 messages, enabling communication with hospital information systems (HIS) and radiology information systems (RIS), where required.

These libraries ensure compliance with established healthcare-interoperability standards while supporting local, offline operation.

### Version Control

Git was selected as the Version Control System (VCS) because of its reliability and lightweight design. These characteristics make it well suited to managing diverse research outputs, including datasets, source code, figures, and notes.

Git provides features for tracking and comparing versions, retracing errors, and maintaining a comprehensive audit trail (RAM, 2013). It has also become a widely adopted medium for sharing and maintaining scholarly code (ESCAMILLA et al., 2022).

GitHub was selected as the Git Hosting Platform (GHP) because of its stability, broad compatibility with research workflows, and ability to centralize versioned materials in an accessible environment.
