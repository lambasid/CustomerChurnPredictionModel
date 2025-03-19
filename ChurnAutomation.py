import pandas as pd
import mysql.connector
import smtplib
from email.mime.text import MIMEText
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Connect to SQL Database
conn = mysql.connector.connect(host="localhost", user="root", password="password", database="customer_db")
query = "SELECT * FROM customer_data"
df = pd.read_sql(query, conn)
conn.close()

# 2️⃣ Data Preprocessing
df.fillna(0, inplace=True)
X = df[['Age', 'Subscription_Length_Months', 'Monthly_Spend', 'Support_Tickets']]
y = df['Churn']

# 3️⃣ Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 4️⃣ Make Predictions
df['Churn_Prediction'] = model.predict(X)

# 5️⃣ Save Predictions to Excel
df[['Customer_ID', 'Churn_Prediction']].to_excel("Churn_Predictions.xlsx", index=False)
print("Churn predictions saved to Excel.")

# 6️⃣ Send Alert Emails for High-Risk Customers
high_risk_customers = df[df['Churn_Prediction'] == 1]
if not high_risk_customers.empty:
    msg_body = "High-risk customers detected:\n" + high_risk_customers[['Customer_ID', 'Monthly_Spend']].to_string()
    msg = MIMEText(msg_body)
    msg["Subject"] = "Customer Retention Alert"
    msg["From"] = "business@company.com"
    msg["To"] = "support@company.com"

    server = smtplib.SMTP("smtp.office365.com", 587)
    server.starttls()
    server.login("your_email", "password")
    server.sendmail(msg["From"], msg["To"], msg.as_string())
    server.quit()
    print("Alert email sent!")

print("Automation completed successfully.")
