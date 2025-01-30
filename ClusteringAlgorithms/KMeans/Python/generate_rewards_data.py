import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of records to generate
n_records = 100000

# Generate Customer IDs
customer_ids = np.random.randint(1001, 2001, size=n_records)

# Generate Product IDs
product_ids = np.random.randint(1, 11, size=n_records)

# Generate New Reward Percentages (with some potential anomalies)
new_rewards = np.random.normal(0.10, 0.02, size=n_records)  # Mean 10%, std dev 2%
new_rewards = np.clip(new_rewards, 0.01, 0.25)  # Ensure values are within a reasonable range

# Introduce some anomalies (e.g., 5% of the data)
anomaly_indices = np.random.choice(n_records, size=int(n_records * 0.05), replace=False)
new_rewards[anomaly_indices] = np.random.uniform(0.30, 0.50, size=len(anomaly_indices)) # Anomaly values

# Generate Renew Reward Percentages (correlated with new rewards, with some noise)
renew_rewards = new_rewards * 0.95 + np.random.normal(0, 0.01, size=n_records) # Slightly lower, with noise
renew_rewards = np.clip(renew_rewards, 0.01, 0.25)

# Generate Eligibility Start and End Dates
start_dates = [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
end_dates = [start + timedelta(days=np.random.randint(30, 365)) for start in start_dates]


# Create a Pandas DataFrame
data = pd.DataFrame({
    'CustomerId': customer_ids,
    'ProductId': product_ids,
    'NewRewardPercentage': new_rewards,
    'RenewRewardPercentage': renew_rewards,
    'EligibilityStartDate': start_dates,
    'EligibilityEndDate': end_dates
})

# Convert dates to string format for SQL compatibility (optional)
data['EligibilityStartDate'] = data['EligibilityStartDate'].dt.strftime('%Y-%m-%d')
data['EligibilityEndDate'] = data['EligibilityEndDate'].dt.strftime('%Y-%m-%d')


# Save to CSV (or you can directly use this DataFrame in your Python code)
data.to_csv('rewards_data.csv', index=False)

print(data.head()) # Display the first few rows