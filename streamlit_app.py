import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
from ipywidgets import interact
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("🎈Business Intelligence (BI)")

# Create a sidebar for navigation
st.sidebar.header("Menu")

def main():
    # Define options for the selectbox
    options = ["Home", "Data Analysis", "Visualization", "Predictions", "Pivot Table"]

    # Create a selectbox in the sidebar
    selected_option = st.sidebar.selectbox("Choose an option:", options)

    # Display content based on user selection
    if selected_option == "Home":
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://media.istockphoto.com/id/1189708364/photo/abstract-white-background.jpg?b=1&s=612x612&w=0&k=20&c=luc9srapqCloKOthp3ncQySU0vf5ULcQ9exjKPjruy8=");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.write("Data Analysis of Manhattan Rental Prices!")
        
        st.markdown(
        """ 
        Data analysis of Manhattan rental prices to potential customers, the goal is to provide actionable insights that help them make informed decisions. 
        
        Here are some key ideas to focus on:

        1. Market Trends and Forecasting:

        Historical Price Trends: Visualize how rental prices have fluctuated over time, highlighting seasonal variations and long-term trends.
        
        Predictive Analytics: Use forecasting models to project future price movements, helping clients anticipate market shifts.
        
        Neighborhood Analysis: Compare price trends across different neighborhoods, identifying areas with potential for higher returns or lower risk.
        
        2. Pricing Strategies:

        Optimal Pricing: Analyze factors influencing rental prices (e.g., size, amenities, location) to determine optimal pricing strategies.
        
        Competitive Analysis: Benchmark against competitors' pricing and identify opportunities for differentiation.
        
        Sensitivity Analysis: Explore how changes in market conditions (e.g., interest rates, economic indicators) might impact rental prices.
        
        
        3. Investment Opportunities:
        

        High-Yield Properties: Identify properties with high potential for rental income and capital appreciation.
        
        Risk Assessment: Evaluate the risks associated with different investment strategies, such as vacancy rates and market volatility.
        
        Portfolio Optimization: Provide recommendations for diversifying investment portfolios to mitigate risk.
        
        
        4. Tenant Insights:
        

        Tenant Demographics: Analyze the characteristics of the tenant population (e.g., age, income, occupation) to tailor marketing and leasing strategies.
        
        Tenant Preferences: Understand tenant preferences for amenities, location, and lease terms to optimize property offerings.
        
        
        5. Interactive Data Visualization:

        Interactive Dashboards: Create interactive dashboards that allow users to explore data at different levels of detail.
        
        Customizable Reports: Provide customizable reports tailored to specific needs and preferences.
        
        Data Storytelling: Use storytelling techniques to communicate complex insights in a clear and engaging manner.
        
        By focusing on these key areas, we can provide valuable insights that help customers make informed decisions about their Manhattan rental investments.
                
        
        """)

    elif selected_option == "Data Analysis":
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://images.pexels.com/photos/12977021/pexels-photo-12977021.jpeg?auto=compress&cs=tinysrgb&w=800=");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.subheader("Data Analysis: Rental Price", divider=True)        
        NY = "https://raw.githubusercontent.com/adanque/RentalPricePrediction/refs/heads/main/Datasets/renthopNYC.csv"
        df = pd.read_csv(NY)
        st.write(df)
        
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

        st.subheader("Descriptive Statistics", divider=True)
        pd.options.display.float_format = '{:,.0f}'.format
        st.write(df['price'].describe())
        
        guess = df['price'].mean()
        errors = guess - df['price']
        mean_absolute_error = errors.abs().mean()
        st.write(f'If we just guessed every RentHop NY condo Rent for ${guess:,.0f},')
        st.write(f'we would be off by ${mean_absolute_error:,.0f} on average.')


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
        
        
        st.subheader("Map: Price by Bedrooms", divider=True)
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

    elif selected_option == "Visualization":
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://images.pexels.com/photos/12977021/pexels-photo-12977021.jpeg?auto=compress&cs=tinysrgb&w=800=");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.subheader("Visualize Multiple Variables.", divider=True)
        st.markdown(
        """ 
        In this scatter plot, you can analyze different perspectives from the dataset by selecting various variables based on the information you need to examine.
        """)
        NY = "https://raw.githubusercontent.com/adanque/RentalPricePrediction/refs/heads/main/Datasets/renthopNYC.csv"
        df = pd.read_csv(NY)
            
        # Filter DataFrame 
        df = df[(df['price'] >= 1375) & (df['price'] <= 15500) & 
                (df['latitude'] >= 40.57) & (df['latitude'] < 40.99) &
                (df['longitude'] >= -74.1) & (df['longitude'] <= -73.38)]

        # Create a scatter plot using Plotly Express
        fig = px.scatter(df, x='longitude', y='latitude', opacity=0.2,
                            title='Scatter Plot of NYC Apartments',
                            labels={'longitude': 'Longitude', 'latitude': 'Latitude'})
        
        grouped = df.groupby(['bathrooms', 'bedrooms', 'created', 'display_address',
                            'latitude', 'longitude', 'street_address', 'interest_level']).agg({'price': 'sum'}).reset_index()

        # Add a dropdown to select the x-axis column
        x_axis_column = st.selectbox('Select 1 option', grouped.columns)

        # Add a dropdown to select the y-axis column
        y_axis_column = st.selectbox('Select 2 option', grouped.columns)

        # Create the Plotly figure
        fig = px.scatter(grouped, x=x_axis_column, y=y_axis_column, title='Interactive Scatter Plot')
        
        # Customize the figure (optional)
        fig.update_layout(
            xaxis_title=x_axis_column,
            yaxis_title=y_axis_column
        )
        # Display the figure in Streamlit
        st.plotly_chart(fig)
        

    elif selected_option == "Predictions":
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://images.pexels.com/photos/12977021/pexels-photo-12977021.jpeg?auto=compress&cs=tinysrgb&w=800=");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.subheader("Data Analysis: Rental Price", divider=True)  
        from sklearn.metrics import mean_absolute_error
        st.subheader("Model Predictions", divider=True)
        NY = "https://raw.githubusercontent.com/adanque/RentalPricePrediction/refs/heads/main/Datasets/renthopNYC.csv"
        df = pd.read_csv(NY)
            
        # Filter DataFrame 
        df1 = df[(df['price'] >= 1375) & (df['price'] <= 15500) & 
                (df['latitude'] >= 40.57) & (df['latitude'] < 40.99) &
                (df['longitude'] >= -74.1) & (df['longitude'] <= -73.38)]
        
        df = pd.DataFrame(df1)

        # Train a simple linear regression model (replace this with your actual model)
        model = LinearRegression()
        model.fit(df[['bedrooms']], df['price'])

        # Prediction function
        def predict(bedrooms):
            y_pred = model.predict([[bedrooms]])
            estimate = y_pred[0]
            coefficient = model.coef_[0]
            
            # Format with $ and comma separators. No decimals.
            result = f'Rent for a {bedrooms}-bedroom apartment in New York City is estimated at ${estimate:,.0f}.'
            explanation = f' Each additional bedroom is associated with a ${coefficient:,.0f} increase in this model.'
            return result + '\n' + explanation

        # Add a dropdown to select the number of bedrooms
        bedroom_selection = st.selectbox('Select number of bedrooms:', df['bedrooms'].unique())

        # Display prediction result
        prediction_result = predict(bedroom_selection)
        st.write(prediction_result)

        # Create an interactive scatter plot of the original data
        fig = px.scatter(df, x='bedrooms', y='price', title='Apartment Prices vs. Bedrooms',
                        labels={'bedrooms': 'Number of Bedrooms', 'price': 'Price'},
                        hover_data=['price'])

        # Display the figure in Streamlit
        st.plotly_chart(fig)

        st.subheader("Price Statistic", divider=True)
        m = model.coef_[0]
        b  = model.intercept_
        # st.write(f'y = {m:,.0f}*x + {b:,.0f}')
        # st.write(f'price = {m:,.0f} * bedrooms + {b:,.0f}')
        st.write(df['price'].describe())
        
        # 2. Instantiate this class
        model = LinearRegression()

        # 3. Arrange x features matrix & y target vector
        features = ['bedrooms']
        target = 'price'
        X= df[features]
        y = df[target]

        #4. fit model
        model.fit(X,y)
        type(df[['bedrooms']])
        type(df['price'])
        features = ['bedrooms']
        target ='price'
        X_train = df[features]
        y_train = df[target]
        # 4. fit the model
        model.fit(X_train, y_train)
        # 5. Apply the model to new data
        bedrooms = 1000
        X_test = [[bedrooms]]
        y_pred = model.predict(X_test)
        y_pred
        y_test = [3500]
        # 5. Apply the model to *new/unknown* data
        def predict(bedrooms):
            y_pred = model.predict([[bedrooms]])
            estimate = y_pred[0]
            coefficient = model.coef_[0]

            # Format with $ and comma separators. No decimals.
            result = f'Rent for a {bedrooms}-bedroom apartment in New York City is estimated at ${estimate:,.0f}.'
            explanation = f' Each additional bedroom is associated with a ${coefficient:,.0f} increase in this model.'
            return result + explanation

        predict(1)
        High_rentG = [6000]
        Medium_rentG = [4500]
        Low_rentG = [3000]
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f'Our model Mean Absolute Error: ${mae:,.0f}')
        mae = mean_absolute_error(y_test, High_rentG)
        st.write(f'High Rent MAE: ${mae:,.0f}')
        mae = mean_absolute_error(y_test, Medium_rentG)
        st.write(f'Medium Rent MAE: ${mae:,.0f}')
        mae = mean_absolute_error(y_test, Low_rentG)
        st.write(f'Low Rent MAE: ${mae:,.0f}')
        
        
        df = pd.DataFrame(df)

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(df[['bedrooms']], df['price'])

        # Define the prediction function with error metrics
        def predict(bedrooms, actual_price=None):
            y_pred = model.predict([[bedrooms]])
            estimate = y_pred[0]
            coefficient = model.coef_[0]
            
            # Format the results
            result = f'Rent for a {bedrooms}-bedroom apartment is estimated at ${estimate:,.0f}.'
            explanation = f'Each additional bedroom is associated with a ${coefficient:,.0f} increase.'
            
            if actual_price is not None:
                mae = mean_absolute_error([actual_price], y_pred)
                result += f'\nMean Absolute Error (MAE): ${mae:,.0f}'
            
            return result + '\n' + explanation

        # Streamlit UI components for interactivity
        st.title("Apartment Price Prediction")

        # Slider for selecting number of bedrooms (similar to interact)
        bedroom_selection = st.slider('Select number of bedrooms:', min_value=1, max_value=5)

        # Input for actual price comparison
        actual_price = st.number_input('Enter the actual rent price for comparison:', min_value=0)

        # Display prediction result based on user input
        if actual_price > 0:
            prediction_result = predict(bedroom_selection, actual_price)
        else:
            prediction_result = predict(bedroom_selection)

        st.write(prediction_result)
        
        # Create a 3D scatter plot
        fig = px.scatter_3d(df, x='bedrooms', y='bathrooms', z='price', opacity=0.5,
                            title='3D Scatter Plot of Apartment Prices',
                            labels={'bedrooms': 'Number of Bedrooms', 
                                    'bathrooms': 'Number of Bathrooms', 
                                    'price': 'Price'})

        # Display the plot in Streamlit
        st.title("Apartment Price Prediction")
        st.plotly_chart(fig)   
        
        
    elif selected_option == "Pivot Table":
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("https://images.pexels.com/photos/12977021/pexels-photo-12977021.jpeg?auto=compress&cs=tinysrgb&w=800=");
        background-size: cover;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
        st.subheader("Data Analysis: Rental Price", divider=True)  
        from matplotlib import pyplot as plt
        st.subheader("Pivot Table", divider=True)
        NY = "https://raw.githubusercontent.com/adanque/RentalPricePrediction/refs/heads/main/Datasets/renthopNYC.csv"
        df = pd.read_csv(NY)
            
        # Filter DataFrame 
        df = df[(df['price'] >= 1375) & (df['price'] <= 15500) & 
                (df['latitude'] >= 40.57) & (df['latitude'] < 40.99) &
                (df['longitude'] >= -74.1) & (df['longitude'] <= -73.38)]
        
        df = pd.DataFrame(df)

        # Create a pivot table
        table = df.pivot_table(values='price', index='bedrooms', columns='bathrooms', aggfunc='mean')

        # Set up the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(table, annot=True, fmt=',.0f', cbar=True)

        # Display the heatmap in Streamlit
        st.title("Apartment Price Heatmap")
        st.write("Heatmap of average prices based on bedrooms and bathrooms:")
        st.pyplot(plt)

        # Optionally display the pivot table as well
        st.write("Pivot Table:")
        

if __name__ == "__main__":
   main()