            f'<div class="result-card">'
            f'<div class="result-title">'
            f'Predicted Risk Score'
            f'</div>'
            f'<div class="score">'
            f'{risk_score:.4f}'
            f'</div>'
            f'<div class="{category_class}">'
            f'{risk_category}'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">Input Summary</div>',
            unsafe_allow_html=True
        )

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            left_summary = (
                f'<div class="info-card">'
                f'<b>City:</b> {city}<br><br>'
                f'<b>Road Type:</b> {road_type}<br><br>'
                f'<b>Weather:</b> {weather}<br><br>'
                f'<b>Visibility:</b> {visibility}<br><br>'
                f'<b>Traffic Density:</b> {traffic_density}'
                f'</div>'
            )

            st.markdown(
                left_summary,
                unsafe_allow_html=True
            )


        with summary_col2:

            right_summary = (
                f'<div class="info-card">'
                f'<b>Date:</b> '
                f'{selected_date.strftime("%Y-%m-%d")}<br><br>'
                f'<b>Day:</b> {day_of_week}<br><br>'
                f'<b>Time:</b> '
                f'{selected_time.strftime("%H:%M")}<br><br>'
                f'<b>Weekend:</b> {weekend_text}<br><br>'
                f'<b>Peak Hour:</b> {peak_text}'
                f'</div>'
            )

            st.markdown(
                right_summary,
                unsafe_allow_html=True
            )


        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        st.markdown(
            '<div class="section-title">Explainable AI</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-card">'
            'SHAP explains how the input features contributed '
            'to the predicted risk score.'
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        with st.spinner(
            "Generating explanation..."
        ):

            explainer = create_shap_explainer(
                model
            )

            shap_values = calculate_shap_values(
                explainer,
                processed_input
            )


        if shap_values is not None:

            feature_names = get_feature_names(preprocessor)

            values = shap_values[0]

            if len(feature_names) == len(values):

                shap_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "SHAP Value": values,
                        "Absolute Impact": np.abs(values)
                    }
                )

                shap_df = shap_df.sort_values(
                    "Absolute Impact",
                    ascending=False
                )


                top_features = (
                    shap_df
                    .head(10)
                    .sort_values(
                        "Absolute Impact",
                        ascending=True
                    )
                )


                # ------------------------------------------------
                # SHAP CHART
                # ------------------------------------------------

                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                ax.barh(
                    top_features["Feature"],
                    top_features["SHAP Value"]
                )

                ax.set_xlabel(
                    "SHAP Value"
                )

                ax.set_ylabel(
                    "Feature"
                )

                ax.set_title(
                    "Top Features Contributing to Risk Prediction"
