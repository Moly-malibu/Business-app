import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("🎈Pharmaceutical Business")

# Create a sidebar for navigation
st.sidebar.header("Menu")

def main():
    # Define options for the selectbox
    options = ["Home", "Data Analysis", "Visualization", "Settings"]

    # Create a selectbox in the sidebar
    selected_option = st.sidebar.selectbox("Choose an option:", options)

    # Display content based on user selection
    if selected_option == "Home":
        st.write("Welcome to the Home page!")

    elif selected_option == "Data Analysis":
        st.write("Here you can analyze your data.")
        
        NY = "https://raw.githubusercontent.com/adanque/RentalPricePrediction/refs/heads/main/Datasets/renthopNYC.csv"
        df = pd.read_csv(NY)
        st.write(df)
        df.shape == (49352, 34)
        
        # Filter DataFrame 
        df = df[(df['price'] >= 1375) & (df['price'] <= 15500) & 
                (df['latitude'] >= 40.57) & (df['latitude'] < 40.99) &
                (df['longitude'] >= -74.1) & (df['longitude'] <= -73.38)]

        # Create a scatter plot using Plotly Express
        fig = px.scatter(df, x='longitude', y='latitude', opacity=0.2,
                        title='Scatter Plot of NYC Apartments',
                        labels={'longitude': 'Longitude', 'latitude': 'Latitude'})

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.title("Descriptive Statistics")
        pd.options.display.float_format = '{:,.0f}'.format
        st.write(df['price'].describe())


        # #Create a distribution plot for the 'price' column
        df = df[(df['price'] >= 1375) & (df['price'] <= 15500)]

        # Create an interactive histogram using Plotly Express
        fig = px.histogram(df, x='price', nbins=30, title='Distribution of Apartment Prices in NYC',
                        labels={'price': 'Price'},
                        opacity=0.75)

        # Add a line for the kernel density estimate (KDE)
        kde_fig = px.density_contour(df, x='price', title='KDE of Apartment Prices in NYC')
        fig.add_trace(kde_fig.data[0])  # Add KDE trace to histogram

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        #first baseline
        guess = df['price'].mean()
        errors = guess - df['price']
        mean_absolute_error = errors.abs().mean()
        st.write(f'If we fust guessed the price would be: ${guess:,.0f}')
        st.write(f'we would be off by ${mean_absolute_error:,.0f} on average.')


        # Create an interactive scatter plot using Plotly Express
        fig = px.scatter(df, x='bedrooms', y='price', 
                        title='Scatter Plot of Apartment Prices vs. Bedrooms',
                        labels={'bedrooms': 'Number of Bedrooms', 'price': 'Price'},
                        hover_data=['price'],  # Show price on hover
                        opacity=0.7)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Create an interactive scatter plot using Plotly Express
        fig = px.scatter(df, x='longitude', y='price', 
                        title='Scatter Plot of Apartment Prices vs. Longitude',
                        labels={'longitude': 'Longitude', 'price': 'Price'},
                        hover_data=['bedrooms'],  # Show number of bedrooms on hover
                        opacity=0.7)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Create an interactive scatter map using Plotly Express
        fig = px.scatter_mapbox(df, 
                                lat='latitude', 
                                lon='longitude', 
                                color='price',  # Color by price for better visualization
                                hover_name='bedrooms',  # Show number of bedrooms on hover
                                hover_data={'price': True, 'bedrooms': True},  # Additional data on hover
                                title='Apartment Prices in NYC',
                                color_continuous_scale=px.colors.sequential.Viridis,  # Change to a different color scale
                                size_max=15,
                                zoom=10)

        # Update layout for Mapbox
        fig.update_layout(mapbox_style="open-street-map")  # Use OpenStreetMap style for free access

        # Display the map in Streamlit
        st.plotly_chart(fig, use_container_width=True)


        # Create an interactive scatter plot with a trendline using Plotly Express
        fig = px.scatter(df, 
                        x='bedrooms', 
                        y='price', 
                        title='Scatter Plot of Apartment Prices vs. Bedrooms with Trendline',
                        labels={'bedrooms': 'Number of Bedrooms', 'price': 'Price'},
                        trendline='ols',  # Add ordinary least squares trendline
                        hover_data={'price': True, 'bedrooms': True},  # Show additional data on hover
                        opacity=0.7)

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # st.markdown(
        # """ 
        # In this scatter plot, you can analyze different perspectives from the dataset by selecting various variables based on the information you need to examine

        # """)
        # grouped = df.groupby(['bathrooms', 'bedrooms', 'created', 'display_address',
        #                     'latitude', 'longitude', 'street_address', 'interest_level']).agg({'price': 'sum'}).reset_index()

        # # Add a dropdown to select the x-axis column
        # x_axis_column = st.selectbox('Select 1 option', grouped.columns)

        # # Add a dropdown to select the y-axis column
        # y_axis_column = st.selectbox('Select 2 option', grouped.columns)

        # # Create the Plotly figure
        # fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')
        # # Customize the figure (optional)
        # fig.update_layout(
        #     xaxis_title=x_axis_column,
        #     yaxis_title=y_axis_column
        # )
        # # Display the figure in Streamlit
        # st.plotly_chart(fig)


        st.title("PREDICTIONS")
        #1. library scikit learn
        # Sample DataFrame creation 
        # data = {
        #     'bedrooms': [1, 2, 3, 4, 5],
        #     'price': [1500, 2500, 3500, 4500, 6000]
        # }
        # df = pd.DataFrame(data)

        # # Train a simple linear regression model (replace this with your actual model)
        # model = LinearRegression()
        # model.fit(df[['bedrooms']], df['price'])

        # # Prediction function
        # def predict(bedrooms):
        #     y_pred = model.predict([[bedrooms]])
        #     estimate = y_pred[0]
        #     coefficient = model.coef_[0]
            
        #     # Format with $ and comma separators. No decimals.
        #     result = f'Rent for a {bedrooms}-bedroom apartment in New York City is estimated at ${estimate:,.0f}.'
        #     explanation = f' Each additional bedroom is associated with a ${coefficient:,.0f} increase in this model.'
        #     return result + explanation

        # # Add a dropdown to select the number of bedrooms
        # # bedroom_selection = st.selectbox('Select number of bedrooms:', df['bedrooms'].unique())

        # # Display prediction result
        # prediction_result = predict(bedroom_selection)
        # st.write(prediction_result)

        # # Create an interactive scatter plot of the original data
        # fig = px.scatter(df, x='bedrooms', y='price', title='Apartment Prices vs. Bedrooms',
        #                 labels={'bedrooms': 'Number of Bedrooms', 'price': 'Price'},
        #                 hover_data=['price'])

        # # Display the figure in Streamlit
        # st.plotly_chart(fig)

        # st.title("The error for our prediction")

        
        # Sample DataFrame creation
        data = {
            'bedrooms': [1, 2, 3, 4, 5],
            'price': [1500, 2500, 3500, 4500, 6000]
        }
        df = pd.DataFrame(data)

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(df[['bedrooms']], df['price'])

        # Define the prediction function with error metrics
        def predict(bedrooms, actual_price=None):
            y_pred = model.predict([[bedrooms]])
            estimate = y_pred[0]
            coefficient = model.coef_[0]
            
            result = f'Rent for a {bedrooms}-bedroom apartment is estimated at ${estimate:,.0f}.'
            explanation = f'Each additional bedroom is associated with a ${coefficient:,.0f} increase.'
            
            if actual_price is not None:
                mae = mean_absolute_error([actual_price], y_pred)
                mse = mean_squared_error([actual_price], y_pred)
                r2 = r2_score([actual_price], y_pred)
                
                result += f'\nMean Absolute Error (MAE): ${mae:,.0f}'
                result += f'\nMean Squared Error (MSE): ${mse:,.0f}'
                result += f'\nR-squared: {r2:.2f}'
            
            return result + '\n' + explanation

        # Streamlit UI components
        bedroom_selection = st.selectbox('Select number of bedrooms:', df['bedrooms'].unique())
        actual_price = st.number_input('Enter the actual rent price for comparison:', min_value=0)

        # Display prediction result
        if actual_price > 0:
            prediction_result = predict(bedroom_selection, actual_price)
        else:
            prediction_result = predict(bedroom_selection)

        st.write(prediction_result)

        def predict_with_metrics(bedrooms, actual_price=None):
            y_pred = model.predict([[bedrooms]])
            estimate = y_pred[0]
            coefficient = model.coef_[0]
            
            # Prepare the results dictionary
            results_dict = {
                'Estimated Rent': [estimate],
                'Coefficient per Bedroom': [coefficient],
            }
            
            if actual_price is not None:
                mae = mean_absolute_error([actual_price], y_pred)
                mse = mean_squared_error([actual_price], y_pred)
                r2 = r2_score([actual_price], y_pred)
                
                results_dict.update({
                    'Mean Absolute Error (MAE)': [mae],
                    'Mean Squared Error (MSE)': [mse],
                    'R-squared': [r2],
                })
            
            # Convert to DataFrame for better display
            results_df = pd.DataFrame(results_dict)
            
            return results_df

        # Display prediction results in a table
        prediction_results_df = predict_with_metrics(bedroom_selection, actual_price if actual_price > 0 else None)
        st.dataframe(prediction_results_df)

        import plotly.express as px

        def visualize_metrics(mae, mse, r2):
            metrics_data = {
                'Metric': ['MAE', 'MSE', 'R-squared'],
                'Value': [mae, mse, r2]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig = px.bar(metrics_df, x='Metric', y='Value', title='Model Performance Metrics')
            st.plotly_chart(fig)

        # Example usage after calculating metrics
        if actual_price > 0:
            mae = mean_absolute_error([actual_price], y_pred)
            mse = mean_squared_error([actual_price], y_pred)
            r2 = r2_score([actual_price], y_pred)
            
            visualize_metrics(mae, mse, r2)
       







    # elif selected_option == "Visualization":
    #     st.write("Visualize your data here.")
    #     # Additional code for visualization features can go here


    # elif selected_option == "Settings":
    #     st.write("Adjust your settings here.")
    #     # Additional code for settings features can go here


    # # Optional: Add a form for additional inputs

    # with st.form(key='my_form'):
    #     input_data = st.text_input("Enter some data:")
    #     submit_button = st.form_submit_button("Submit")
        
    #     if submit_button:
    #         st.write(f"You entered: {input_data}")

if __name__ == "__main__":
   main()