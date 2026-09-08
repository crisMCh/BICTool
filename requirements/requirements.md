#  Requirements

In this chapter, the requirements derived from the work package descriptions are presented in a tabular requirements specification. Functional and non-functional requirements are listed with their identifiers, descriptions, and priorities to provide a clear overview of the system’s expected behavior and constraints.

##  General Requirements

| Section | Description |
|---|---|
| **Purpose** | Define functional, non-functional, data, interface, and validation requirements for the BICTool software that provides objective, automated, and quantitative image quality assessment in head CT images. |
| **Scope** | The BICTool is an open-source, standalone, and modular software tool capable of analyzing DICOM-based head CT images, calculating quantitative image quality metrics, such as MTF, NPS, and PS/MTF, correlating them with radiation dose indicators, including CTDIvol and DLP, and providing standardized visual and tabular output for the optimization of CT protocols. |
| **Stakeholders** | Bundesamt für Strahlenschutz (BfS), radiologists, radiographers, medical physicists, clinicians, software developers, and researchers. |
| **Intended Use** | Continuous and standardized assessment of CT image quality for clinical and research use, dose optimization, and comparison with human observer studies. |

##  System Context

| Aspect | Description |
|---|---|
| **System Environment** | Runs as a standalone desktop application and interacts with PACS/RIS systems via DICOM/HL7. |
| **Users** | Radiologists, medical physicists, researchers, and IT administrators. |
| **Assumptions** | Users have valid access to anonymized DICOM data, and data collection complies with GDPR and institutional data policies. |
| **Dependencies** | Python/C++ environment, DICOM libraries (pydicom for dataset handling, pynetdicom for the DICOM network protocol used to communicate with PACS), AI frameworks (TensorFlow/PyTorch), Git version control, PACS interface standards. |

## Functional Requirements

| ID | Requirement Description | Type | Priority |
|---|---|---|---|
| **FR-01** | The system shall import DICOM CT images, including header metadata such as CTDIvol and DLP. | Functional | High |
| **FR-02** | The system shall automatically detect and segment anatomical structures, such as the skull, brain, ventricles, and vessels. | Functional | High |
| **FR-03** | The system shall compute quantitative image quality metrics, including Modulation Transfer Function (MTF), Noise Power Spectrum (NPS), Power Spectrum to MTF ratio (PS/MTF), and other task-based metrics. | Functional | High |
| **FR-04** | The system shall provide a quality score per image slice and an overall study-level rating. | Functional | High |
| **FR-05** | The system shall be impartial to reconstructed images from different algorithms, including filtered back projection (FBP), iterative reconstruction, and AI-based reconstruction, for fidelity assessment, i.e., metric computation should not be biased by which reconstruction method produced the image. | Functional | High |
| **FR-06** | The system shall allow the user to visualize segmented structures, metrics, and noise maps through an interactive GUI. | Functional | Medium |
| **FR-07** | The system shall export results in Excel, ASCII, or DICOM-compatible formats for reporting and integration. | Functional | Medium |
| **FR-08** | The system shall include a plugin system with a well-defined, in-process Python API and hook mechanism for extending analysis modules, ensuring encapsulation, backward compatibility, and documented integration guidelines for plugin developers. | Functional | High |
| **FR-09** | The system shall log user activity and processing details for traceability and quality assurance. | Functional | Medium |
| **FR-10** | The system shall enable human observer study integration for comparative validation (WP4). | Functional | Medium |
| **FR-11** | The system may support AI-based denoising or other preprocessing before segmentation. | Functional | Medium |

## Non-Functional Requirements

| ID | Requirement Description | Category | Priority |
|---|---|---|---|
| **NFR-01** | The system shall comply with DICOM and HL7 version 2.x standards for data exchange. | Interoperability | High |
| **NFR-02** | The tool shall follow a modular architecture with independent modules for preprocessing, segmentation, feature extraction, and visualization. | Architecture | High |
| **NFR-03** | The tool shall operate as open-source software under an approved license, such as GPLv3. | Legal | High |
| **NFR-04** | The software shall maintain compatibility across Windows, macOS, and Linux systems. | Portability | Medium |
| **NFR-05** | The average processing time per CT study containing 100 slices shall not exceed a defined time on standard workstation hardware. | Performance | Medium |
| **NFR-06** | The system shall ensure secure handling of de-identified data in accordance with GDPR and BDSG. | Security | High |
| **NFR-07** | The GUI shall adhere to usability heuristics, including Nielsen’s principles, to ensure an intuitive user experience. | Usability | Medium |
| **NFR-08** | Version control shall be maintained using Git, enabling continuous integration and continuous delivery (CI/CD). | Maintainability | High |
| **NFR-09** | All algorithms shall be validated against at total of at least 10 test CT patients from three different reconstruction algorithms. | Validation | High |
| **NFR-10** | The tool shall provide documentation and user manuals describing input/output, parameters, and troubleshooting procedures. | Documentation | High |