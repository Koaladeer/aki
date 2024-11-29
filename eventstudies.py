import pandas as pd
import matplotlib.pyplot as plt
from eventstudy.eventStudy import EventStudy
import eventstudy as es

class EventStudyAnalysis:
    def __init__(self, study_file, stock_file):
        self.study_file = study_file
        self.stock_file = stock_file
        self.study_data = None
        self.stock_data = None
        self.merged_data = None

    def load_data(self):
        """Loads study and stock data from CSV files."""
        self.study_data = pd.read_csv(self.study_file)
        self.stock_data = pd.read_csv(self.stock_file)

    def preprocess_stock_data(self):
        """Preprocesses stock data by formatting dates and calculating returns."""
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'], format='%m/%d/%Y')
        self.stock_data['Change %'] = self.stock_data['Change %'].str.replace('%', '').astype(float) / 100.0
        self.stock_data = self.stock_data.sort_values(by='Date')

    def preprocess_study_data(self):
        """Preprocesses study data by formatting dates and filtering relevant columns."""
        self.study_data['Primary Completion Date'] = pd.to_datetime(
            self.study_data['Primary Completion Date'], errors='coerce'
        )
        self.study_data['Completion Date'] = pd.to_datetime(self.study_data['Completion Date'], errors='coerce')
        self.study_data['Results First Posted'] = pd.to_datetime(
            self.study_data['Results First Posted'], errors='coerce'
        )
        self.study_data = self.study_data[
            ['Primary Completion Date', 'Completion Date', 'Results First Posted', 'Enrollment']
        ]

    def filter_events(self):
        """Filters events and prepares a DataFrame for the event study."""
        events = []
        for event_type in ['Primary Completion Date', 'Completion Date', 'Results First Posted']:
            event_subset = self.study_data[['Enrollment', event_type]].rename(columns={event_type: 'Date'})
            event_subset = event_subset.dropna().assign(EventType=event_type)
            events.append(event_subset)
        events_df = pd.concat(events, ignore_index=True)

        # Merge events with stock data
        self.merged_data = pd.merge_asof(
            events_df.sort_values('Date'), self.stock_data.sort_values('Date'), on='Date', direction='backward'
        )
        self.merged_data['EventID'] = range(len(self.merged_data))

    def perform_event_study(self):
        """
        Performs the event study for multiple event types using the market model and returns the results for each type.
        Returns:
            dict: A dictionary where keys are event types and values are the event study results.
        """
        import os

        # Create a temporary directory for CSV files
        temp_dir = 'temp_event_study'
        os.makedirs(temp_dir, exist_ok=True)

        event_types = self.merged_data['EventType'].unique()  # Unique event types in the dataset
        results = {}

        for event_type in event_types:
            # Filter events for the current type
            events = self.merged_data[self.merged_data['EventType'] == event_type]

            # Create a CSV file for this event type
            event_csv_path = os.path.join(temp_dir, f'{event_type}_events.csv')
            events[['Date', 'Enrollment']].to_csv(
                event_csv_path, index=False, date_format='%d/%m/%Y', header=['event_date', 'enrollment']
            )

            # Perform the event study using the Multiple API
            event_study_result = es.Multiple.from_csv(
                path=event_csv_path,
                event_study_model=es.Single.market_model,  # Use market model for the study
                event_window=(-5, +10),  # Analyze 5 days before and 10 days after the event
                estimation_size=200,  # Estimation window size
                buffer_size=30,  # Buffer period size
                date_format='%d/%m/%Y',
                ignore_errors=True
            )

            # Store results for this event type
            results[event_type] = event_study_result

        return results

    @staticmethod
    def plot_event_study(results, title):
        """Plots AAR and CAR results."""
        plt.figure(figsize=(10, 6))

        # AAR
        plt.subplot(2, 1, 1)
        plt.plot(results['time_window'], results['AAR'], label='AAR', marker='o')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f'{title} - AAR')
        plt.xlabel('Days Relative to Event')
        plt.ylabel('AAR (%)')
        plt.legend()

        # CAR
        plt.subplot(2, 1, 2)
        plt.plot(results['time_window'], results['CAR'], label='CAR', marker='o', color='orange')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f'{title} - CAR')
        plt.xlabel('Days Relative to Event')
        plt.ylabel('CAR (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()


def main():
    # Initialize the EventStudyAnalysis class
    study_file = "Data\studies_data_v2.csv"
    stock_file = "Data\stock_data.csv"
    esa = EventStudyAnalysis(study_file, stock_file)

    # Load and preprocess data
    esa.load_data()
    esa.preprocess_stock_data()
    esa.preprocess_study_data()

    # Filter events and perform event study
    esa.filter_events()
    results = esa.perform_event_study()

    # Plot the results
    esa.plot_event_study(results, 'Event Study Analysis')


if __name__ == "__main__":
    main()
