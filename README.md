# Multi-Label Email Classification System

This project implements a multi-label email classification system that can classify emails across multiple interdependent dimensions (Type 2, Type 3, and Type 4) while maintaining proper architectural design principles.

## Features

- **Modular Architecture**: Separates preprocessing, embedding, and modeling components
- **Consistent Input Data**: Maintains consistent data format across all models
- **Uniform Model Interface**: Provides consistent access to different models via abstraction
- **Multi-Label Classification**: Supports two different approaches:
  - **Design Choice 1**: Chained Multi-Output Classification
  - **Design Choice 2**: Hierarchical Modeling

## Project Structure

```
Actvity 3 Full Solution/
├── data/
│   ├── AppGallery.csv       - Customer support tickets for Huawei's AppGallery
│   └── Purchasing.csv       - Purchase/subscription related support tickets
├── model/
│   ├── base.py              - Abstract base class for models
│   ├── randomforest.py      - Original single-label RandomForest model
│   ├── chainedforest.py     - Multi-label chained RandomForest implementation (Design Choice 1)
│   └── hierarchicalforest.py - Hierarchical RandomForest implementation (Design Choice 2)
├── modelling/
│   ├── data_model.py        - Data encapsulation with hierarchical support
│   └── modelling.py         - Model training and evaluation logic with approach selection
├── Config.py                - Configuration settings with approach selection
├── embeddings.py            - Text embedding generation
├── main.py                  - Main controller
├── preprocess.py            - Data preprocessing functions
```

## Architecture Design

This implementation follows software engineering principles to create a modular architecture:

### 1. Separation of Concerns
The system is divided into distinct components:
- Data preprocessing (preprocess.py)
- Feature embedding (embeddings.py)
- Data encapsulation (data_model.py)
- Model implementation (model/*.py)
- Configuration (Config.py)
- Main controller (main.py)

Each component has a specific responsibility and can be modified independently.

### 2. Data Encapsulation
The Data class in data_model.py encapsulates all data-related functionality:
- Maintains consistent data format across models
- Handles train/test splitting
- Provides specialized data structures for both classification approaches
- Exposes a consistent interface for accessing data

### 3. Abstraction
The BaseModel abstract class defines a uniform interface for all models:
- Ensures consistent methods across implementations
- Forces implementation of required functionality
- Allows interchangeable model usage

## Multi-Label Classification Approaches

### Design Choice 1: Chained Multi-Output Classification

This approach uses a single model instance for each label combination:

1. **Model Structure**:
   - One model for Type 2 classification
   - One model for Type 2 + Type 3 classification
   - One model for Type 2 + Type 3 + Type 4 classification

2. **Label Representation**:
   - Combined labels using a separator (e.g., "Problem/Fault_+_Payment_+_Subscription cancellation")
   - Each combination is treated as a distinct class

3. **Advantages**:
   - Simpler implementation with fewer models
   - Consistent data usage across all classification tasks
   - Directly models label dependencies

4. **Disadvantages**:
   - The number of possible combined classes grows exponentially
   - Each combined class has fewer training examples
   - Cannot leverage hierarchical structure as directly

### Design Choice 2: Hierarchical Modeling

This approach creates multiple model instances in a tree-like structure:

1. **Model Structure**:
   - Top level: One model for Type 2 classification 
   - Middle level: Separate models for each Type 2 class to predict Type 3
   - Bottom level: Separate models for each Type 2 + Type 3 combination to predict Type 4

2. **Prediction Flow**:
   - First predict Type 2 using the top-level model
   - Use that prediction to select the appropriate Type 3 model
   - Use both predictions to select the appropriate Type 4 model

3. **Advantages**:
   - Models specialize in specific contexts
   - Follows the natural hierarchy in the data
   - Each model only needs to distinguish between a smaller set of classes

4. **Disadvantages**:
   - More complex implementation with many models to manage
   - Smaller training sets for deeper levels in the hierarchy
   - Errors can cascade (an error at a higher level affects all predictions below it)

## Understanding Class Imbalance and Zeros in Results

Both approaches may show many classes with zero precision, recall, and F1-scores in the classification reports. This is **not** a bug but a natural consequence of:

1. **Data Sparsity**: The datasets are relatively small (200 samples total) divided across many classes
2. **Class Imbalance**: Many classes have very few examples (0-3 samples)
3. **Multi-Label Complexity**: As we add more label dimensions, the number of distinct combinations grows exponentially
4. **Hierarchical Dependencies**: Later variables (Type 3, Type 4) depend on earlier ones (Type 2)

This leads to several expected patterns:

- Type 2 classification typically has the highest accuracy (0.40-0.93)
- Type 2 + Type 3 has medium accuracy (0.16-0.73)
- Type 2 + Type 3 + Type 4 has the lowest accuracy (0.12-0.67)

These patterns confirm the assessment brief's prediction that "the accuracy of the latter variable will definitely be less or equal to 80% and can't be more than that."

## Configuration

The system can be configured using the `Config.py` file:

```python
# Enable/disable multi-label classification
ENABLE_MULTI_LABEL = True

# Choose the model approach
MODEL_APPROACH = "both"  # Options: "chained", "hierarchical", or "both"

# Define label combinations
LABEL_COMBINATIONS = [
    ['y2'],             # Type 2 only
    ['y2', 'y3'],       # Type 2 + Type 3
    ['y2', 'y3', 'y4']  # Type 2 + Type 3 + Type 4
]
```

## Running the System

### Prerequisites
- Python 3.6+
- Required libraries: numpy, pandas, scikit-learn

### Installation
1. Clone or download this repository
2. Ensure the data files are in the `data/` directory

### Execution
1. Configure the system in `Config.py` (see Configuration section)
2. Run the main script:
   ```
   python main.py
   ```
3. The system will:
   - Load and preprocess data
   - Generate embeddings
   - Create appropriate data models
   - Train the selected approach(es)
   - Generate predictions
   - Evaluate and display results

### Example Output
The system will display:
- Classification reports for each label combination
- Accuracy summaries for each approach
- A comparison table if both approaches are run

## Interpreting Results

When analyzing results, consider:

1. **Overall Accuracy**: Higher values indicate better performance
2. **Per-Class Metrics**: Many zeros are expected due to data limitations
3. **Approach Comparison**: The hierarchical approach often performs better for Type 2 + Type 3 but may perform worse for Type 2 + Type 3 + Type 4 due to cascading errors

## Extending the System

To extend the system:
- Add new model types by creating classes that inherit from BaseModel
- Add more datasets in the data/ directory
- Modify Config.py to include additional configuration options
- Enhance preprocessing.py for better text cleaning and normalization

## Team

This project was implemented as part of the Extreme Programming Continuous Assessment for Enterprise Architecture Design.

## License

This project is available for educational purposes. 
