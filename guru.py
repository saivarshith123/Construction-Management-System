import oracledb
from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Establish connection to Oracle database
connection = oracledb.connect(user="system", password="abc", dsn="localhost:1521/XE")
cursor = connection.cursor()

@app.route('/projects')
def get_projects():
    cursor.execute('SELECT * FROM project')
    projects = cursor.fetchall()
    
    return jsonify(projects)

@app.route('/<pid>')
def get_project(pid):
    cursor.execute("SELECT * FROM project WHERE pid={}".format(pid))
    project = cursor.fetchall()
    cursor.execute('SELECT * FROM division WHERE pid={}'.format(pid))
    division = cursor.fetchall()
    data = {
        "project": project,
        "division": division
    }
    return jsonify(data)

@app.route("/register2", methods=["POST"])
def register2():
    pid = request.form["pid"]

    cursor.execute("SELECT pid FROM project WHERE pid='{}'".format(pid))
    validate = cursor.fetchall()

    if len(validate) == 0:
        cursor.execute("INSERT INTO project(pid) VALUES({})".format(pid))
        connection.commit()
        return jsonify({"message": "Project registered successfully."}), 201
    else:
        return jsonify({"error": "Project with given PID already exists."}), 400

@app.route("/register3", methods=["POST"])
def register3():
        pid = request.form.get("pid")
        division_no = request.form.get("division_no")
        no_workers = request.form.get("no_workers")

        # Check if all required fields are present
        if pid is None or division_no is None or no_workers is None:
            return jsonify({"error": "Missing required fields"}), 400

        # Perform any necessary validation on the input data

        # Insert the data into the database
        cursor.execute("INSERT INTO division (pid, division_no, no_workers) VALUES ({},{},{})".format(pid, division_no, no_workers))
        connection.commit()

        return jsonify({"message": "Division registered successfully."}), 201
@app.route("/delete_division", methods=["POST"])
def delete_division():
    # Parse input data from the request
    pid = request.form.get("pid")
    division_no = request.form.get("division_no")

    # Check if all required fields are present
    if pid is None or division_no is None:
        return jsonify({"error": "Missing required fields"}), 400

    # Execute the DELETE SQL query
    cursor.execute("DELETE FROM division WHERE pid = {} AND division_no = {}".format(pid, division_no))
    connection.commit()

    return jsonify({"message": "Division deleted successfully."}), 200

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r'C:\Users\rajuv\construction_materials_dataset.csv'
data = pd.read_csv(file_path)

# Check the first few rows of the dataset
print(data.head())

# Features and target
X = data[['land_area(sq. meters)', 'number_of_floors', 'number_of_rooms', 'House_Design']]
y = data[['Bricks(units)', 'Cement(bags)', 'Steel(kg)', 'Wood(cubic meters)']]

# One-hot encoding for categorical feature (House_Design)
categorical_features = ['House_Design']
numeric_features = ['land_area(sq. meters)', 'number_of_floors', 'number_of_rooms']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Random Forest model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f"Mean Squared Errors for Bricks, Cement, Steel, Wood: {mse}")

@app.route('/<land_area_sq_meters>-<number_of_floors>-<number_of_rooms>-<House_Design>')
def pridict_values(land_area_sq_meters,number_of_floors,number_of_rooms,House_Design):
    new_input = pd.DataFrame({
        'land_area(sq. meters)': [land_area_sq_meters], 
        'number_of_floors': [number_of_floors], 
        'number_of_rooms': [number_of_rooms], 
        'House_Design': [House_Design]
    })
    new_prediction = model.predict(new_input)
    print((list(new_prediction[0])))
    k=list(new_prediction[0])
    return k

def main():
    land_area_sq_meters=500
    number_of_floors=10
    number_of_rooms=15
    House_Design='Colonial'
    pridict_values(land_area_sq_meters,number_of_floors,number_of_rooms,House_Design)
    





# Handling requests for favicon.ico
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')





if __name__ == '__main__':
    main()
    app.run(debug=True, host='0.0.0.0')
