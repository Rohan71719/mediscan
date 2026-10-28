### Performance & Resource Footprint

#### System Environment
* **Framework:** PyTorch `2.11.0+cpu`
* **Hardware:** 2 Logical CPUs

####  Latency Benchmarks
*Measurements include image reading, preprocessing steps, and inference (SLA Target Limit: **200 ms**).*

| Worker Threads | Median Latency | 95th Percentile | Max Latency | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1 Thread** | 21.9 ms | 23.8 ms | 27.8 ms |  Within SLA |
| **2 Threads** | 21.5 ms | 23.9 ms | 24.9 ms |  Optimal Performance |
| **4 Threads** | 22.9 ms | 25.0 ms | 27.7 ms |  Diminishing Returns (CPU Over-subscription) |

####  Model Artifact Storage
*Total footprint evaluation against deployment size boundaries.*

* **`arogyanet.pt`:** 1.27 MB (Model Weights)
* **`gate_stats.npz`:** 0.26 MB (OOD Gating Calibration Statistics)
* **`config.json`:** < 0.01 MB (Configuration File)
* **Total Directory Size:** **1.54 MB** / 25.00 MB Maximum Limit (6.16% capacity utilized)
