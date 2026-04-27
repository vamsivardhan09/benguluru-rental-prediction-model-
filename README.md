# Bangalore Rental Prediction Model

A machine learning project to predict rental prices for properties in Bangalore based on property features and location data.

## 📊 Dataset

The project uses rental property data from 8 localities in Bangalore:
- Bellandur
- Brookefield
- Electronic City
- K.R Puram
- Kaggadasapura
- Varthur
- Whitefield
- Yelahanka

**Total Properties:** 2,572
**Features:** Property size, type, furnishing, amenities, location, lease type, etc.

## 🧠 Machine Learning Model

- **Algorithm:** Linear Regression
- **Performance:**
  - R² Score: 0.59
  - Mean Absolute Error: ₹4,548
  - Root Mean Squared Error: ₹6,560

## 🚀 Web Application

Interactive Streamlit app for rent prediction: `app.py`

### Features:
- Input property details via user-friendly interface
- Real-time rent predictions
- Model performance metrics display

### How to Run:
```bash
pip install streamlit scikit-learn pandas joblib
streamlit run app.py
```

## 📁 Project Structure

```
├── pandasclass.ipynb          # Jupyter notebook with analysis and model training
├── app.py                     # Streamlit web application
├── rent_prediction_model.pkl  # Trained model file
├── model_features.pkl         # Model feature information
├── *.csv                      # Dataset files for each locality
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## 🔧 Requirements

- Python 3.8+
- pandas
- scikit-learn
- streamlit
- joblib
- matplotlib

## 📈 Key Insights

### Top Rent Influencing Factors:
1. **Property Type:** BHK4PLUS properties command highest rents
2. **Lease Type:** Company leases are most expensive
3. **Location:** Bellandur has premium pricing
4. **Size:** Larger properties cost more
5. **Amenities:** Gym and swimming pool add value

### Average Rents by Locality:
- **Highest:** Whitefield (~₹25,000)
- **Lowest:** K.R Puram (~₹12,000)

## 🎯 Usage

1. **For Analysis:** Open `pandasclass.ipynb` in Jupyter
2. **For Prediction:** Run `streamlit run app.py`
3. **For Development:** Load model with `joblib.load('rent_prediction_model.pkl')`

## 🤝 Contributing

Feel free to fork this repository and contribute improvements!

## 📄 License

This project is open source. Please check individual dataset licenses if applicable.