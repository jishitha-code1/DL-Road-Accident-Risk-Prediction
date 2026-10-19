                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Weather", inferred_weather.title())
                    m2.metric("Traffic", inferred_traffic.title())
                    m3.metric("Visibility", inferred_visibility.title())
                    m4.metric("Potholes", pothole_count)

                    if vehicle_counts:
                        st.write("Vehicle counts:", vehicle_counts)
                    st.caption(
                        "Weather estimate: "
                        f"{analysis.get('weather_label', 'unknown')} "
                        f"(classifier confidence {analysis.get('weather_confidence', 0.0):.1%}). "
                        "A single image can misclassify weather and only shows visible vehicles."
                    )

                    # Use the existing trained risk model with inferred image features.
                    # Current time is used because time cannot reliably be inferred from a still image.
                    inference_time = datetime.now().time().replace(second=0, microsecond=0)
                    image_processed_input = preprocess_input(
                        weather=inferred_weather,
                        visibility=inferred_visibility,
                        traffic_density=inferred_traffic,
                        selected_time=inference_time,
                        is_peak_hour=calculate_peak_hour(inference_time.hour),
                        preprocessor=preprocessor,
                        expected_features=EXPECTED_FEATURES,
                    )
                    base_score = predict_risk(model, image_processed_input)

                    # Potholes were not a trained input feature, so incorporate them as
                    # a transparent experimental adjustment rather than pretending the
                    # DNN learned their effect. Must be validated before scientific use.
                    pothole_adjustment = (
                        0.0 if pothole_count == 0
                        else min(0.25, 0.10 + 0.04 * (pothole_count - 1))
                    )
                    combined_score = float(np.clip(
                        base_score + pothole_adjustment * (1.0 - base_score), 0.0, 1.0
                    ))
                    combined_category = get_risk_category(combined_score)
                    category_class = {
                        "Low Risk": "low-risk",
                        "Medium Risk": "medium-risk",
                        "High Risk": "high-risk",
                    }[combined_category]

                    st.markdown("### Combined accident-risk estimate")
                    st.markdown(
                        f'<div class="result-card"><div class="result-title">'
                        f'Image-Based Combined Risk Score</div>'
                        f'<div class="score">{combined_score:.4f}</div>'
                        f'<div class="{category_class}">{combined_category}</div></div>',
                        unsafe_allow_html=True,
                    )
                    s1, s2, s3 = st.columns(3)
                    s1.metric("DNN score from inferred conditions", f"{base_score:.4f}")
                    s2.metric("Pothole adjustment", f"+{pothole_adjustment:.3f}")
                    s3.metric("Vehicles counted", vehicle_total)

                    st.markdown("### Why this risk was estimated")
                    explanations = []
                    if pothole_count:
                        explanations.append(
                            f"**Potholes:** {pothole_count} possible pothole(s) were detected, "
                            "which may require drivers to brake or change direction."
                        )
                    else:
                        explanations.append(
                            "**Potholes:** no potholes were detected at the current detector threshold; "
                            "this does not prove the road is damage-free."
                        )
                    explanations.append(
                        f"**Weather:** the image classifier estimated **{inferred_weather}** conditions. "
                        "Rain or fog may reduce traction or visibility."
                    )
                    explanations.append(
                        f"**Traffic:** {vehicle_total} car/bus/truck/motorcycle instance(s) were visible, "
                        f"mapped to **{inferred_traffic}** traffic density for this estimate."
                    )
                    explanations.append(
                        f"**Time:** the model used the current time ({inference_time.strftime('%H:%M')}) "
                        "because a single road image cannot reliably reveal the time of day."
                    )
                    for explanation in explanations:
                        st.markdown(f"- {explanation}")

                    st.warning(
                        "Important research limitation: your existing DNN was trained on structured "
                        "conditions, not raw images, and pothole count is not one of its trained inputs. "
                        "The displayed combined score adds a hand-designed pothole adjustment; it is "
                        "not a calibrated probability. Validate this method against labelled data before "
                        "reporting it as an accurate accident-risk predictor."
                    )

        except subprocess.TimeoutExpired:
            st.error("Image analysis exceeded the 10-minute limit. First-time model downloads may be the cause.")
        except Exception as exc:
            st.error("Image-based risk prediction could not be completed.")
            st.exception(exc)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    '<div class="footer-text">DL-Based Road Accident Risk Prediction | '
    'Deep Learning + Explainable AI</div>',
    unsafe_allow_html=True,
)
