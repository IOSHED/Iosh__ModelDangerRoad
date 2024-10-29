import csv
import random
import numpy as np


# Generate a dataset of 1000 records with more complex accident occurrence logic
def generate_data(filename, num_rows=1001):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow([
            "statistics_accidents_on_road", "coating_humidity", "quality_coating", "traffic_congestion",
            "width_road", "allowed_max_speed", "allowed_min_speed", "turns_route", "tangles_turns_on_route",
            "has_marking_applied", "confidence_on_road", "driving_experience", "age", "total_traffic_violations",
            "total_offenses", "number_accidents", "marginality_population", "wind_direction", "wind_speed",
            "accident_occurred"
        ])

        # Helper functions to generate random values for each field
        def random_wind_direction():
            return random.choice(["left", "right", "forward", "front", "after"])

        def random_angles_turns(turns):
            return [random.randint(5, 90) for _ in range(turns)]

        for _ in range(num_rows - 1):
            statistics_accidents_on_road = round(random.uniform(0.1, 0.5), 2)  # 0.1 to 0.5
            coating_humidity = round(random.uniform(0, 0.6), 2)  # 0 to 0.6
            quality_coating = round(random.uniform(0.5, 1.0), 2)  # 0.5 to 1.0
            traffic_congestion = round(random.uniform(0.1, 0.8), 2)  # 0.1 to 0.8
            width_road = random.randint(6, 10)  # 6 to 10m
            allowed_max_speed = random.choice([50, 60, 80, 100, 120])  # Speed limits
            allowed_min_speed = 0
            turns_route = random.randint(2, 10)  # Number of turns
            tangles_turns_on_route = random_angles_turns(turns_route)
            has_marking_applied = random.choice([True, False])
            confidence_on_road = round(random.uniform(0.5, 1), 2)
            driving_experience = random.randint(1, 30)  # Driving experience in years
            age = random.randint(18, 70)  # Driver age
            total_traffic_violations = random.randint(0, 10)
            total_offenses = random.randint(0, 3)
            number_accidents = random.randint(0, 5)
            marginality_population = round(random.uniform(0, 0.5), 2)
            wind_direction = random_wind_direction()
            wind_speed = random.randint(0, 15)

            # Complex logic for accident occurrence
            risk_factors = (
                    statistics_accidents_on_road * 0.4 +
                    coating_humidity * 0.3 +
                    (1 - quality_coating) * 0.2 +
                    traffic_congestion * 0.3 +
                    (10 - driving_experience) * 0.05 +
                    marginality_population * 0.4 +
                    (total_traffic_violations / 10) * 0.2 +
                    (number_accidents / 5) * 0.3
            )

            # Calculate accident occurrence probability
            accident_occurred = int(np.random.rand() < risk_factors)  # Convert probability to binary outcome

            # Write the row to the CSV file
            writer.writerow([
                statistics_accidents_on_road, coating_humidity, quality_coating, traffic_congestion,
                width_road, allowed_max_speed, allowed_min_speed, turns_route, tangles_turns_on_route,
                has_marking_applied, confidence_on_road, driving_experience, age, total_traffic_violations,
                total_offenses, number_accidents, marginality_population, wind_direction, wind_speed, accident_occurred
            ])


# Generate the data and save to a file
generate_data("../dataset/road_accidents_data_with_answer_1.csv")
