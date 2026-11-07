# MediScan: report to the CEO

**Bottom line.** The previous model's "98.4% accuracy" cannot predict how it behaves at a health camp, and
"99% accuracy" is the wrong target. We rebuilt the model, which now finds about 98 of every 100 sick
children at a stated cost of about 83 false alarms per 1,000 children screened, fits in 1.2 MB. It is a screening aid, not a diagnosis, and must not be used on adults.

## 1. What went wrong with the 98.4% (instruction 1: proof)
- The earlier test and validation sets could not support the claim: the validation set had **16** images (a
  perfect score on 16 images is consistent with a true accuracy as low as 79%), and the test set was 62.5%
  pneumonia drawn from one hospital. Nothing was recorded (no code, split or seeds), so it cannot be reproduced.
- We looked for leakage between his training and test folders and **found none** (no shared patient or
  duplicate images; 26 near-duplicate pairs, almost all look-alikes). So the 98.4% is not a leakage artefact.
  It is an in-hospital number: our rebuilt models score 94% (a classical method) to 97% on comparable data.
- The data contain **shortcuts**: three numbers about an image (size, shape, brightness) separate the classes
  with ROC-AUC 0.905, and a model that never sees the lungs' centre scores 0.86 balanced accuracy. A model can
  look excellent on this hospital's images and fail on another hospital's equipment.
- That is what the pilot showed. In our own stress test on images from other sources (predominantly adults),
  the false-alarm rate rose from 9% to 35%. The same failure direction as the pilot's 7 in 10, though the
  populations differ and the sizes are not comparable.
- Accuracy hides this: with 62.5% sick children in his test set, "call everyone sick" already scores 62.5%.

## 2. What the rebuilt model achieves (instruction 2)
ArogyaNet (our own network, 300,641 parameters, trained from scratch). Fixed before testing: threshold 0.416.
Evaluated once on 852 locked images (238 healthy, 614 sick):

| | Result | 95% interval |
|---|---|---|
| Sick children detected (sensitivity) | 98.4% | 97.0 to 99.2 |
| Healthy children correctly cleared (specificity) | 90.8% | 86.3 to 94.1 |
| Accuracy | 96.2% | 95.0 to 97.5 |

**At the camp (about 1 sick child per 9 healthy), per 1,000 children:** about 98 of 100 sick children are
flagged and 1.6 missed; about 83 healthy children are flagged too, so only **54% of flagged children are
actually sick**. Cross-validation and the validation split agree (specificity 91% to 93%).
The tool also says "low confidence, refer to a clinician" for probabilities between 0.42 and 0.90 and refuses non-X-ray images: it refused blank, noise and colour images and a greyscale dog photo; one greyscale cat photo was not refused (it was sent to clinician review), and about 1% of genuine X-rays (13 of 1,000 tested) are refused and need a retake

## 3. Why "99% accuracy" is the wrong target
Per 1,000 camp children (100 sick, 900 healthy), a miss costs about 20 times a false alarm:

| Strategy | Accuracy | Sick missed | False alarms | Cost (20 × missed + false alarms) |
|---|---|---|---|---|
| Refer nobody | 90% | 100 | 0 | 2,000 |
| Hypothetical 99% model, all 10 errors are misses | 99% | 10 | 0 | 200 |
| Hypothetical 99% model, all 10 errors are false alarms | 99% | 0 | 10 | 10 |
| **Rebuilt ArogyaNet (locked test)** | **91.5%** | **1.6** | **83** | **116** (interval 77 to 154) |

A tool that screens no one is already 90% accurate, and two models with the same 99% accuracy differ twentyfold
in harm. Our model has lower accuracy than a hypothetical "99%" model that misses ten children, yet it is
better for the children. Reaching 99% at the camp would need fewer than about 8 false alarms per 1,000 healthy
children (specificity above 99%); our model achieves 91%, and no model we built exceeds 95% on this data.
**Ask for this instead:** sensitivity of at least 98% with the lower end of the 95% interval above 95%,
specificity of at least 90%, and expected cost per 1,000 children, all measured with confidence intervals on
paediatric images from an independent hospital at the camp's prevalence.

## 4. Speed and size 
Model size 1.3 MB (1.5 MB with the safety checks), limit 25 MB. CPU time per image on a 2-logical-CPU environment: median 18.5 to 22.9 ms, 95th percentile 21.8 to 25.0 ms, worst single call 49.8 ms (200 runs after warm-up; 2 logical CPUs, two sessions, one to four threads; limit 200 ms; includes reading and preprocessing but not start-up of a fresh process)


## 5. What it must not be used for
- As a diagnosis, or to rule pneumonia out in a child who has symptoms.
- On adults or children outside 1 to 5 years (false alarms of 35% on other-source images).
- On other scanners or hospitals without re-measuring performance first.
- Without a clinician reviewing every referral.
Limits of our evidence: single-hospital training data; the "external" dataset overlapped our data (2,686 of 11,537
images were copies) and its independent part had no pneumonia cases, so sensitivity at a new hospital has **not**
been measured. A prospective pilot is needed before deployment.
