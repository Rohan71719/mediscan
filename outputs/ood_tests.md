### Out-of-Distribution (OOD) Validation Log

####  Image Rejection Test Results

| Input Image / Type | Format / Variant | Status | Reason / Metrics |
| :--- | :--- | :--- | :--- |
| **blank white** | — |  **REJECTED** | Blank or almost uniform image |
| **blank black** | — |  **REJECTED** | Blank or almost uniform image |
| **uniform grey** | — |  **REJECTED** | Blank or almost uniform image |
| **noise** | Grey |  **REJECTED** | $d^2 = 2146$ (Does not resemble target chest X-rays) |
| **noise** | Colour |  **REJECTED** | Colour photograph, not a greyscale chest X-ray |
| **gradient** | Grey |  **REJECTED** | $d^2 = 563$ (Does not resemble target chest X-rays) |
| **photo chelsea** | Colour |  **REJECTED** | Colour photograph, not a greyscale chest X-ray |
| **photo chelsea** | Greyscale |  **LOW CONFIDENCE** | Refer to a clinician ($p = 0.792$, $d^2 = 387$) |
| **photo astronaut** | Colour |  **REJECTED** | Colour photograph, not a greyscale chest X-ray |
| **photo astronaut** | Greyscale |  **REJECTED** | $d^2 = 1609$ (Does not resemble target chest X-rays) |
| **photo coffee** | Colour |  **REJECTED** | Colour photograph, not a greyscale chest X-ray |
| **photo coffee** | Greyscale |  **REJECTED** | $d^2 = 696$ (Does not resemble target chest X-rays) |
| **my dog.jpg** | Colour |  **REJECTED** | Colour photograph, not a greyscale chest X-ray |
| **my dog.jpg** | Greyscale | ❌ **REJECTED** | $d^2 = 634$ (Does not resemble target chest X-rays) |

####  System Edge Cases & Error Metrics

* **Leaked / False Negatives (Failures):** 
  * `['photo chelsea, greyscale']` — This image bypassed hard rejection and fell back to a low confidence warning.
* **Primary Dataset Rejection Rate:** 
  * **2.7%** (8 out of 300 real chest X-rays from the mixed train/val/test split were incorrectly rejected).
