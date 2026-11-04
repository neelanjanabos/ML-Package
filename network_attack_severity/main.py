import os
import json
import pandas as pd

from preprocessing import (
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    get_sample_presets
)

from predict import SeverityPredictor

from database import (
    init_db,
    log_prediction,
    get_prediction_history,
    get_history_summary,
    clear_prediction_history
)


predictor = SeverityPredictor()

METRICS_PATH = os.path.join("models", "metrics.json")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def pause():
    input("\nPress Enter to continue...")


def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}

    try:
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    except Exception:
        return {}


def check_models():
    if not os.path.exists("models"):
        return False

    required_files = [
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "knn.pkl",
        "naive_bayes.pkl",
        "svm.pkl",
        "attack_classifier.pkl",
        "scaler.pkl",
        "encoders.pkl"
    ]

    for file in required_files:
        if not os.path.exists(os.path.join("models", file)):
            return False

    return True


def show_header():
    print("=" * 60)
    print("       NETWORK ATTACK SEVERITY PREDICTION SYSTEM")
    print("=" * 60)


def select_model():
    available_models = list(predictor.models.keys())

    if not available_models:
        print("\nNo trained models are available.")
        print("Please train the models first.")
        return None

    print("\nAvailable Models")
    print("-" * 40)

    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    while True:
        choice = input("\nEnter model number: ")

        try:
            choice = int(choice)

            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]

            print("Invalid choice.")

        except ValueError:
            print("Please enter a number.")


def get_float_input(prompt, default=0):
    while True:
        value = input(f"{prompt} [{default}]: ").strip()

        if value == "":
            return default

        try:
            return float(value)

        except ValueError:
            print("Please enter a valid number.")


def get_text_input(prompt, default=""):
    value = input(f"{prompt} [{default}]: ").strip()

    if value == "":
        return default

    return value


def choose_prediction_input():
    presets = get_sample_presets()

    print("\nInput Method")
    print("-" * 40)
    print("1. Normal traffic sample")
    print("2. Reconnaissance sample")
    print("3. DoS sample")
    print("4. Critical attack sample")
    print("5. Enter values manually")

    while True:
        choice = input("\nEnter choice: ")

        if choice == "1":
            return presets["normal"]["values"].copy()

        elif choice == "2":
            if "reconnaissance" in presets:
                return presets["reconnaissance"]["values"].copy()

            if "recon" in presets:
                return presets["recon"]["values"].copy()

            print("Reconnaissance preset not available.")

        elif choice == "3":
            if "dos" in presets:
                return presets["dos"]["values"].copy()

            if "DoS" in presets:
                return presets["DoS"]["values"].copy()

            print("DoS preset not available.")

        elif choice == "4":
            for name in ["critical", "backdoor", "shellcode", "worm", "worms"]:
                if name in presets:
                    return presets[name]["values"].copy()

            print("Critical preset not available.")

        elif choice == "5":
            return manual_input()

        else:
            print("Invalid choice.")


def manual_input():
    data = {}

    print("\nEnter Network Traffic Features")
    print("-" * 50)

    for feature in FEATURE_COLUMNS:

        if feature in CATEGORICAL_FEATURES:
            data[feature] = get_text_input(feature)

        elif feature in NUMERICAL_FEATURES:
            data[feature] = get_float_input(feature)

        else:
            data[feature] = get_float_input(feature)

    return data


def single_prediction():
    clear_screen()
    show_header()

    if not check_models():
        print("\nTrained models were not found.")
        print("Please use the training option first.")
        pause()
        return

    print("\nSINGLE NETWORK FLOW PREDICTION")
    print("-" * 60)

    model_name = select_model()

    if model_name is None:
        pause()
        return

    input_data = choose_prediction_input()

    print("\nAnalyzing network traffic...")
    
    try:
        result = predictor.predict_single(
            input_data,
            model_name=model_name
        )

        print("\n" + "=" * 60)
        print("                 PREDICTION RESULT")
        print("=" * 60)

        print(f"Model       : {result['model_used']}")
        print(f"Severity    : {result['predicted_severity']}")
        print(f"Attack Type : {result['predicted_attack']}")

        confidence = result.get("confidence")

        if confidence is not None:
            print(f"Confidence   : {float(confidence):.2f}%")

        print("=" * 60)

        log_prediction(
            model_used=result["model_used"],
            severity=result["predicted_severity"],
            attack=result["predicted_attack"],
            confidence=result.get("confidence"),
            features=input_data
        )

        print("\nPrediction saved to database.")

    except Exception as e:
        print("\nPrediction failed.")
        print("Error:", str(e))

    pause()


def batch_prediction():
    clear_screen()
    show_header()

    if not check_models():
        print("\nTrained models were not found.")
        print("Please train the models first.")
        pause()
        return

    print("\nBATCH CSV PREDICTION")
    print("-" * 60)

    file_path = input("Enter CSV file path: ").strip()

    if not os.path.exists(file_path):
        print("\nFile not found.")
        pause()
        return

    if not file_path.lower().endswith(".csv"):
        print("\nPlease provide a CSV file.")
        pause()
        return

    model_name = select_model()

    if model_name is None:
        pause()
        return

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            print("\nThe CSV file is empty.")
            pause()
            return

        print("\nProcessing network flows...")

        df_result, severity_distribution, attack_distribution = predictor.predict_batch(
            df,
            model_name=model_name
        )

        os.makedirs("output", exist_ok=True)

        output_path = os.path.join(
            "output",
            "network_attack_predictions.csv"
        )

        df_result.to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print("              BATCH PREDICTION RESULT")
        print("=" * 60)

        print(f"Model used       : {model_name}")
        print(f"Flows processed  : {len(df_result)}")

        print("\nSeverity Distribution")
        print("-" * 30)

        for severity, count in severity_distribution.items():
            print(f"{severity:<15} : {count}")

        print("\nAttack Distribution")
        print("-" * 30)

        for attack, count in attack_distribution.items():
            print(f"{attack:<15} : {count}")

        print("\nPrediction file saved to:")
        print(output_path)

        print("\nSaving first 20 predictions to history...")

        for _, row in df_result.head(20).iterrows():

            log_prediction(
                model_used=model_name,
                severity=row.get(
                    "Predicted_Severity",
                    "UNKNOWN"
                ),
                attack=row.get(
                    "Predicted_Attack",
                    "UNKNOWN"
                ),
                confidence=row.get(
                    "Confidence_%",
                    0.0
                ),
                features=row.to_dict()
            )

        print("History updated.")

    except Exception as e:
        print("\nBatch prediction failed.")
        print("Error:", str(e))

    pause()


def compare_models():
    clear_screen()
    show_header()

    print("\nMODEL COMPARISON")
    print("-" * 60)

    metrics = get_metrics()

    if not metrics:
        print("\nmetrics.json was not found.")
        print("Please train the models first.")
        pause()
        return

    models = metrics.get("models", {})

    if not models:
        print("\nNo model performance data found.")
        pause()
        return

    print(
        f"{'Model':<22}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1 Score':<12}"
    )

    print("-" * 70)

    best_model = metrics.get("best_model")

    for name, data in models.items():

        accuracy = data.get("accuracy", 0)
        precision = data.get("precision", 0)
        recall = data.get("recall", 0)
        f1 = data.get("f1_score", 0)

        print(
            f"{name:<22}"
            f"{accuracy:<12.2f}"
            f"{precision:<12.2f}"
            f"{recall:<12.2f}"
            f"{f1:<12.2f}"
        )

    print("\n" + "-" * 70)

    if best_model:
        print(f"Best Model : {best_model}")

    best_accuracy = metrics.get("best_accuracy")

    if best_accuracy is not None:
        print(f"Best Accuracy : {best_accuracy:.2f}%")

    pause()


def view_history():
    clear_screen()
    show_header()

    print("\nPREDICTION HISTORY")
    print("-" * 60)

    history = get_prediction_history(limit=50)

    if not history:
        print("\nNo prediction history available.")
        pause()
        return

    print(
        f"{'ID':<5}"
        f"{'Model':<20}"
        f"{'Severity':<12}"
        f"{'Attack':<15}"
        f"{'Confidence':<12}"
    )

    print("-" * 70)

    for row in history:

        prediction_id = row.get("id", "")
        model = row.get("model", "")
        severity = row.get("severity", "")
        attack = row.get("attack", "")
        confidence = row.get("confidence", 0)

        try:
            confidence_text = f"{float(confidence):.2f}%"
        except Exception:
            confidence_text = str(confidence)

        print(
            f"{str(prediction_id):<5}"
            f"{str(model)[:18]:<20}"
            f"{str(severity)[:10]:<12}"
            f"{str(attack)[:13]:<15}"
            f"{confidence_text:<12}"
        )

    pause()


def view_summary():
    clear_screen()
    show_header()

    print("\nPREDICTION SUMMARY")
    print("-" * 60)

    summary = get_history_summary()

    print(f"\nTotal Predictions: {summary.get('total', 0)}")

    print("\nSeverity Distribution")
    print("-" * 30)

    severity_dist = summary.get("severity_dist", {})

    if severity_dist:
        for severity, count in severity_dist.items():
            print(f"{severity:<15} : {count}")
    else:
        print("No data.")

    print("\nAttack Distribution")
    print("-" * 30)

    attack_dist = summary.get("attack_dist", {})

    if attack_dist:
        for attack, count in attack_dist.items():
            print(f"{attack:<15} : {count}")
    else:
        print("No data.")

    print("\nModel Usage")
    print("-" * 30)

    model_dist = summary.get("model_dist", {})

    if model_dist:
        for model, count in model_dist.items():
            print(f"{model:<20} : {count}")
    else:
        print("No data.")

    pause()


def clear_history():
    clear_screen()
    show_header()

    print("\nCLEAR PREDICTION HISTORY")
    print("-" * 60)

    summary = get_history_summary()

    print(
        f"\nCurrent number of predictions: "
        f"{summary.get('total', 0)}"
    )

    if summary.get("total", 0) == 0:
        print("There is no history to clear.")
        pause()
        return

    confirm = input(
        "\nType YES to permanently clear history: "
    ).strip()

    if confirm == "YES":
        clear_prediction_history()
        print("\nPrediction history cleared successfully.")

    else:
        print("\nOperation cancelled.")

    pause()


def train_models():
    global predictor

    clear_screen()
    show_header()

    print("\nMODEL TRAINING")
    print("-" * 60)

    confirm = input(
        "\nThis will train all five models.\n"
        "Continue? (Y/N): "
    ).strip().upper()

    if confirm != "Y":
        print("\nTraining cancelled.")
        pause()
        return

    try:
        from train_models import train_and_evaluate

        print("\nTraining models...")
        print("This may take some time.\n")

        train_and_evaluate()

        print("\nTraining completed.")

        print("\nReloading trained models...")

        predictor = SeverityPredictor()

        print("Models reloaded successfully.")

    except Exception as e:
        print("\nModel training failed.")
        print("Error:", str(e))

    pause()


def main_menu():
    init_db()

    while True:

        clear_screen()
        show_header()

        print("\n1. Train Models")
        print("2. Compare Models")
        print("3. Predict Network Traffic")
        print("4. Batch Prediction from CSV")
        print("5. View Prediction History")
        print("6. View Prediction Summary")
        print("7. Clear Prediction History")
        print("8. Exit")

        print("\n" + "-" * 60)

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            train_models()

        elif choice == "2":
            compare_models()

        elif choice == "3":
            single_prediction()

        elif choice == "4":
            batch_prediction()

        elif choice == "5":
            view_history()

        elif choice == "6":
            view_summary()

        elif choice == "7":
            clear_history()

        elif choice == "8":
            print("\nThank you for using the system.")
            print("Exiting...")
            break

        else:
            print("\nInvalid choice.")
            pause()


if __name__ == "__main__":
    main_menu()