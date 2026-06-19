# Uber Bangalore Driver Cancellation Analysis

## Project Overview
This project analyzes 200,000 Uber ride bookings in Bangalore to predict whether a driver will cancel a ride. Through systematic EDA, the dataset was identified as synthetically generated — a finding that was subsequently confirmed by the model's inability to outperform a naive baseline. The project serves as a case study in data quality assessment and the importance of baseline comparison in ML evaluation.

## Dataset
- **Source:** [Kaggle — Uber Ride Details Bangalore](https://www.kaggle.com/datasets/roysouvik98/uber-ride-details-bangalore)
- **Size:** 200,000 rows, 18 columns
- **Key columns:** Booking Status, Pickup Location, Drop Location, Vehicle Type, Avg VTAT, Avg CTAT, Booking Value, Ride Distance, Date, Time

## Key Findings from EDA

### Evidence of synthetic data generation (4 independent signals):
1. **Uniform hourly distribution** — ~8,300 rides per hour across all 24 hours. Real ride data would show clear morning and evening peaks with near-zero activity at 3am.
2. **Uniform cancellation reasons** — all four driver cancellation reasons appear ~9,000 times each. Real data would show uneven distribution.
3. **Uniform cancellation rates by location** — cancellation rates across all 51 Bangalore pickup locations ranged from 28.7% to 32.1% with a standard deviation of only 0.68%. Real data would show significant hotspots.
4. **Uniform cancellation rates by vehicle type** — all 7 vehicle types showed cancellation rates between 30.6% and 31.1%.

## Modeling

### Target variable
Binary classification: 1 = Cancelled by Driver, 0 = all other outcomes (Success, Incomplete, Cancelled by Customer)

### Features used
- Pickup Location, Drop Location (LightGBM native categorical)
- Vehicle Type (Label Encoded)
- Avg VTAT, Avg CTAT, Booking Value, Ride Distance
- Hour of day, Day of week

### Features dropped and why
- `Booking ID`, `Customer ID` — unique identifiers, no predictive value
- `Cancelled Rides by Driver/Customer Reason` — target leakage (only available after cancellation occurs)
- `Driver Ratings`, `Customer Ratings` — target leakage (only exist after ride completion; also unavailable for first-time customers)
- `Payment Method` — only populated for completed rides, leaks the target

### Results
| Model | Accuracy |
|-------|----------|
| Majority class baseline | 82% |
| LightGBM | 81% |

The model performed below the naive baseline — confirming that no meaningful predictive signal exists in this dataset.

### Feature importance
Despite LightGBM assigning highest importance to Drop Location (713) and Pickup Location (703), EDA demonstrated cancellation rates across all 51 locations vary by only 0.68% std. The model was learning noise, not signal.

## Key Conclusion
Synthetic data with uniform distributions across all features provides no learnable signal for a classification model. Accuracy metrics without baseline comparison are misleading — 81% sounds reasonable until you realise a model that always predicts "not cancelled" achieves 82%.

## Limitations and Real World Extension
In real Uber data we would expect:
- Cancellation hotspots at specific pickup locations (airport, outskirts)
- Time-of-day patterns — late night rides more likely to be cancelled
- Vehicle type effects — premium vehicles may have lower cancellation rates
- Driver history and customer rating features would add genuine signal

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, LightGBM, Matplotlib