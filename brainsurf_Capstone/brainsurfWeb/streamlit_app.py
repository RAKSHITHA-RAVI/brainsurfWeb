import streamlit as st
import pandas as pd
import brainsurf.data.csv as csv_import  # Import your Brainsurf library
import numpy as np
import brainsurf.utils.data as util
import brainsurf.preprocessing.filtering as filter
import brainsurf.preprocessing.baseline as baseline
import brainsurf.cognitive_analysis_module.cognitive_comparision as cognitiveStats
from sklearn.model_selection import train_test_split
from brainsurf.machine_learning.eeg_binary_classification import EEGClassifier
from sklearn.metrics import accuracy_score
import brainsurf.data.comparative_visualize as compare
import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with the desired backend

def main():
    st.set_page_config(page_title="Brainsurf IDE")
    st.title("Brainsurf IDE")

    option = st.sidebar.selectbox("Module selection ", ("EEG Analysis","Comparitive Analysis", "Stroop Test Analysis", "Machine Learning Models"))
    if option == "EEG Analysis":
        st.header("EEG Analysis")

        st.subheader("Load EEG Data")

        # Load EEG data from CSV
        file_path = st.text_input("Enter the file path")

        if file_path:
            # Convert CSV to EEGData using your Brainsurf library
            adarsh_pre_med = csv_import.convert_csv_to_eegdata(file_path)
            summary = adarsh_pre_med.summary(10)
            st.write(adarsh_pre_med)
            # Display analysis results
            # st.subheader("Analysis Results")
            # st.write("Summary Data:")
            # st.write(summary)

            values = np.asarray(adarsh_pre_med['sec'], dtype=object)
            st.subheader("Sampling  frequency")
            sampling_freq = util.estimate_sampling_frequency(values)
            st.write(sampling_freq)
            st.sidebar.header("Preprocessing")
            

            artifact_removal = st.sidebar.selectbox("Select a articact removal process",("ICA"))
            if artifact_removal == "ICA":
                st.subheader("ICA")
                
            filter_option = st.sidebar.selectbox("Select a filter", ("Bandpass", "lowpass", "highpass","notch","comb", "adaptive","kalman"))
            if filter_option == "Bandpass":
                st.subheader("Bandpass")
                lowcut = st.number_input("Enter a lowcut", value=0.5)
                highcut = st.number_input("Enter a high cut", value=50)
                order = st.number_input("Enter order", value=4)
                bandpass_filtered_pre_med_eeg = filter.butter_bandpass_filter(adarsh_pre_med['raw'], lowcut, highcut, sampling_freq, order) 
                st.write("Bandpass filtered Data:")
                st.write(bandpass_filtered_pre_med_eeg)
                if st.button('Visualize Unfilered and Filtered data'):
                    factory = compare.ComparativeVisualizationFactory()
                    factory.visualize_multi_feature_time_series(2, adarsh_pre_med['raw'], bandpass_filtered_pre_med_eeg )
            
            # segmentation = st.sidebar.selectbox("Segmentation/epoching",("epoch"))
            # if segmentation == "epochs":
            #     st.subheader("ICA")
            # if segmentation =="":
                # events = eeg_data['event'].values

                # # Extract the EEG signal as a list from the 'raw' column
                # signal = eeg_data['raw'].values.tolist()

                # # Call the create_epochs function to create epochs based on the events
                # epochs = create_epochs(signal, events)

            baseline_correction = st.sidebar.selectbox("Baseline","mean")
            if baseline_correction == "mean":
                st.subheader("Baseline Correction")
                time= adarsh_pre_med["sec"]
                data = np.vstack((time, bandpass_filtered_pre_med_eeg))  # Stack time and filtered EEG signal arrays vertically
                filtered_data_bc = baseline.apply_baseline(data, sampling_freq)
                # Separating the baseline-corrected time and EEG signal
                filtered_eeg_bc = filtered_data_bc[1, :]
                st.write(filtered_data_bc)
            # Normalization

    elif option == "Comparitive Analysis":
        st.header("Comparitive Analysis")
        
        file_path1 = st.text_input("Enter the PRE path")
        file_path2 = st.text_input("Enter the MED path")
        file_path3 = st.text_input("Enter the POST path")
        
        if file_path1 is not None and file_path2 is not None and file_path3 is not None:
            pre_med = csv_import.convert_csv_to_eegdata(file_path1)
            post_med=csv_import.convert_csv_to_eegdata(file_path3)
            med=csv_import.convert_csv_to_eegdata(file_path2)
            pre_med.extract_frequency_bands()
            post_med.extract_frequency_bands()
            pre_med.dropna()
            post_med.dropna()
            min_len = min(len(pre_med), len(post_med))
            pre_no_NA = pre_med[:min_len]
            post_no_NA = post_med[:min_len]
            cognitive_stats = cognitiveStats.analyze_eeg_data(pre_no_NA,post_no_NA)
            st.write(cognitive_stats)

            st.header("Cognitve Analysis")
            
            cog_idx_before, cog_idx_after= cognitiveStats.calculate_cognitive_indexes(pre_no_NA, post_no_NA)
            test_statistic, p_value = cognitiveStats.compare_cognitive_indexes(cog_idx_before, cog_idx_after)
            st.write("test_statistic")
            st.write(test_statistic)
            st.write("p_value")
            st.write(p_value)
            data = {
                'Performance Eval Before': cog_idx_before[0],
                'Arousal Index Before': cog_idx_before[1],
                'Neural Activity Before': cog_idx_before[2],
                'Engagement Before': cog_idx_before[3],
                'Performance Eval After': cog_idx_after[0],
                'Arousal Index After': cog_idx_after[1],
                'Neural Activity After': cog_idx_after[2],
                'Engagement After': cog_idx_after[3]
            }

            df = pd.DataFrame(data)
            st.write(data)

    elif option == "Stroop Test Analysis":
        st.header("Stroop Test Analysis")
        file_path1 = st.text_input("Enter the pre stroop test", key="pre_stroop_test")
        if file_path1:
            stroop_adarsh_pre = pd.read_csv(file_path1)
            st.write(stroop_adarsh_pre)
            st.write(cognitiveStats.analyze_stroop_data(stroop_adarsh_pre))

        file_path2 = st.text_input("Enter the post stroop test", key="post_stroop_test")
        if file_path2:
            stroop_adarsh_post = pd.read_csv(file_path2)
            st.write(stroop_adarsh_post)
            st.write(cognitiveStats.analyze_stroop_data(stroop_adarsh_post))    
    
    elif option == "Machine Learning Models":
        st.header("Machine Learning Module")
        file_pathX = st.text_input("Add the data", key="train")

        if file_pathX:
            data = pd.read_csv(file_pathX)
            X = data[['EEG']].values
            st.write("hi hi")
            y = data['State'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Assuming EEGClassifier is your custom classifier class
            classifier = EEGClassifier()
            classifier.train(X_train, y_train)
            accuracy = classifier.evaluate(X_test, y_test)
            st.write(accuracy)

            st.write("Prediction")
            fileX = st.text_input("Add test data", key="testing")
            if fileX:
                test_data = pd.read_csv(fileX)
                X_test_new = test_data[['EEG']].values
                y_test_new = test_data['State'].values
                new_predictions = classifier.predict(X_test_new)
                st.write(f"New data predictions: {new_predictions}")
                accuracy_new = accuracy_score(y_test_new, new_predictions)
                st.write(f"Accuracy on new data: {accuracy_new}")


       
if __name__ == '__main__':
    main()
