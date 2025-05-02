import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                             QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                             QDialog, QLineEdit, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, silhouette_score
from sklearn.model_selection import KFold, cross_val_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI - Robotics Extension")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_reducer = None

        # Neural network configuration
        self.layer_config = []

        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    # [Previous methods remain the same until create_tabs()]

    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()

        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab),
            ("Robotics Features", self.create_robotics_tab)  # New tab for robotics features
        ]

        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.layout.addWidget(self.tab_widget)

    def create_robotics_tab(self):
        """Create the robotics-specific features tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Data splitting section
        split_group = QGroupBox("Advanced Data Splitting")
        split_layout = QGridLayout()

        # Train-Validation-Test split options
        split_layout.addWidget(QLabel("Train-Validation-Test Split:"), 0, 0)
        self.split_combo = QComboBox()
        self.split_combo.addItems(["70-15-15", "60-20-20", "80-10-10"])
        split_layout.addWidget(self.split_combo, 0, 1)

        # Cross-validation options
        split_layout.addWidget(QLabel("Cross-Validation Folds:"), 1, 0)
        self.cv_spin = QSpinBox()
        self.cv_spin.setRange(2, 20)
        self.cv_spin.setValue(5)
        split_layout.addWidget(self.cv_spin, 1, 1)

        # Apply splitting button
        self.apply_split_btn = QPushButton("Apply Data Splitting")
        self.apply_split_btn.clicked.connect(self.apply_advanced_splitting)
        split_layout.addWidget(self.apply_split_btn, 2, 0, 1, 2)

        split_group.setLayout(split_layout)
        layout.addWidget(split_group)

        # Dimensionality reduction section
        dimred_group = QGroupBox("Advanced Dimensionality Reduction")
        dimred_layout = QVBoxLayout()

        # Method selection
        self.dimred_combo = QComboBox()
        self.dimred_combo.addItems([
            "PCA",
            "Kernel PCA",
            "LDA",
            "t-SNE",
            "UMAP",
            "MDS",
            "Isomap"
        ])
        dimred_layout.addWidget(QLabel("Dimensionality Reduction Method:"))
        dimred_layout.addWidget(self.dimred_combo)

        # Parameters for each method
        self.dimred_params_stack = QStackedWidget()

        # PCA parameters
        pca_widget = QWidget()
        pca_layout = QVBoxLayout(pca_widget)
        self.pca_n_components = QSpinBox()
        self.pca_n_components.setRange(1, 100)
        self.pca_n_components.setValue(2)
        pca_layout.addWidget(QLabel("Number of Components:"))
        pca_layout.addWidget(self.pca_n_components)
        self.dimred_params_stack.addWidget(pca_widget)

        # Kernel PCA parameters
        kpca_widget = QWidget()
        kpca_layout = QVBoxLayout(kpca_widget)
        self.kpca_n_components = QSpinBox()
        self.kpca_n_components.setRange(1, 100)
        self.kpca_n_components.setValue(2)
        self.kpca_kernel = QComboBox()
        self.kpca_kernel.addItems(["linear", "poly", "rbf", "sigmoid", "cosine"])
        kpca_layout.addWidget(QLabel("Number of Components:"))
        kpca_layout.addWidget(self.kpca_n_components)
        kpca_layout.addWidget(QLabel("Kernel:"))
        kpca_layout.addWidget(self.kpca_kernel)
        self.dimred_params_stack.addWidget(kpca_widget)

        # LDA parameters
        lda_widget = QWidget()
        lda_layout = QVBoxLayout(lda_widget)
        self.lda_n_components = QSpinBox()
        self.lda_n_components.setRange(1, 100)
        self.lda_n_components.setValue(2)
        lda_layout.addWidget(QLabel("Number of Components:"))
        lda_layout.addWidget(self.lda_n_components)
        self.dimred_params_stack.addWidget(lda_widget)

        # t-SNE parameters
        tsne_widget = QWidget()
        tsne_layout = QVBoxLayout(tsne_widget)
        self.tsne_n_components = QSpinBox()
        self.tsne_n_components.setRange(1, 3)
        self.tsne_n_components.setValue(2)
        self.tsne_perplexity = QDoubleSpinBox()
        self.tsne_perplexity.setRange(5, 50)
        self.tsne_perplexity.setValue(30)
        tsne_layout.addWidget(QLabel("Number of Components:"))
        tsne_layout.addWidget(self.tsne_n_components)
        tsne_layout.addWidget(QLabel("Perplexity:"))
        tsne_layout.addWidget(self.tsne_perplexity)
        self.dimred_params_stack.addWidget(tsne_widget)

        # UMAP parameters (placeholder)
        umap_widget = QWidget()
        umap_layout = QVBoxLayout(umap_widget)
        umap_layout.addWidget(QLabel("UMAP parameters to be implemented"))
        self.dimred_params_stack.addWidget(umap_widget)

        # MDS parameters
        mds_widget = QWidget()
        mds_layout = QVBoxLayout(mds_widget)
        self.mds_n_components = QSpinBox()
        self.mds_n_components.setRange(1, 3)
        self.mds_n_components.setValue(2)
        mds_layout.addWidget(QLabel("Number of Components:"))
        mds_layout.addWidget(self.mds_n_components)
        self.dimred_params_stack.addWidget(mds_widget)

        # Isomap parameters
        isomap_widget = QWidget()
        isomap_layout = QVBoxLayout(isomap_widget)
        self.isomap_n_components = QSpinBox()
        self.isomap_n_components.setRange(1, 3)
        self.isomap_n_components.setValue(2)
        self.isomap_n_neighbors = QSpinBox()
        self.isomap_n_neighbors.setRange(2, 100)
        self.isomap_n_neighbors.setValue(5)
        isomap_layout.addWidget(QLabel("Number of Components:"))
        isomap_layout.addWidget(self.isomap_n_components)
        isomap_layout.addWidget(QLabel("Number of Neighbors:"))
        isomap_layout.addWidget(self.isomap_n_neighbors)
        self.dimred_params_stack.addWidget(isomap_widget)

        # Connect method selection to parameter display
        self.dimred_combo.currentIndexChanged.connect(self.dimred_params_stack.setCurrentIndex)

        dimred_layout.addWidget(self.dimred_params_stack)

        # Apply button
        self.apply_dimred_btn = QPushButton("Apply Dimensionality Reduction")
        self.apply_dimred_btn.clicked.connect(self.apply_dimensionality_reduction)
        dimred_layout.addWidget(self.apply_dimred_btn)

        dimred_group.setLayout(dimred_layout)
        layout.addWidget(dimred_group)

        # Clustering evaluation section
        cluster_group = QGroupBox("Clustering Evaluation")
        cluster_layout = QVBoxLayout()

        # Elbow method for K-Means
        self.elbow_btn = QPushButton("Run Elbow Method (K-Means)")
        self.elbow_btn.clicked.connect(self.run_elbow_method)
        cluster_layout.addWidget(self.elbow_btn)

        # Silhouette score
        self.silhouette_btn = QPushButton("Calculate Silhouette Score")
        self.silhouette_btn.clicked.connect(self.calculate_silhouette_score)
        cluster_layout.addWidget(self.silhouette_btn)

        cluster_group.setLayout(cluster_layout)
        layout.addWidget(cluster_group)

        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QHBoxLayout()

        self.viz_2d_btn = QRadioButton("2D")
        self.viz_3d_btn = QRadioButton("3D")
        self.viz_2d_btn.setChecked(True)

        viz_button_group = QButtonGroup()
        viz_button_group.addButton(self.viz_2d_btn)
        viz_button_group.addButton(self.viz_3d_btn)

        viz_layout.addWidget(QLabel("Projection:"))
        viz_layout.addWidget(self.viz_2d_btn)
        viz_layout.addWidget(self.viz_3d_btn)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        return widget

    def apply_advanced_splitting(self):
        """Apply train-validation-test split or cross-validation"""
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("No data loaded. Please load a dataset first.")

            # Get selected split strategy
            split_strategy = self.split_combo.currentText()
            cv_folds = self.cv_spin.value()

            if split_strategy == "70-15-15":
                # First split: 70% train, 30% temp
                X_train, X_temp, y_train, y_temp = model_selection.train_test_split(
                    self.X_train, self.y_train, test_size=0.3, random_state=42)
                # Second split: 50-50 of temp (15% each)
                X_val, X_test, y_val, y_test = model_selection.train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42)

                self.X_train, self.y_train = X_train, y_train
                self.X_val, self.y_val = X_val, y_val
                self.X_test, self.y_test = X_test, y_test

                self.status_bar.showMessage(
                    f"Applied {split_strategy} split. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            elif cv_folds > 1:
                # Perform k-fold cross-validation
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                model = LogisticRegression()  # Default model for demonstration

                if len(np.unique(self.y_train)) > 10:  # Regression
                    scores = cross_val_score(model, self.X_train, self.y_train,
                                             cv=kf, scoring='neg_mean_squared_error')
                    scores = -scores  # Convert back to positive MSE
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    self.status_bar.showMessage(
                        f"{cv_folds}-fold CV MSE: {mean_score:.4f} ± {std_score:.4f}")
                else:  # Classification
                    scores = cross_val_score(model, self.X_train, self.y_train,
                                             cv=kf, scoring='accuracy')
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    self.status_bar.showMessage(
                        f"{cv_folds}-fold CV Accuracy: {mean_score:.4f} ± {std_score:.4f}")

                # Update metrics display
                self.metrics_text.setText(
                    f"Cross-Validation Results ({cv_folds}-fold):\n"
                    f"Mean: {mean_score:.4f}\n"
                    f"Std: {std_score:.4f}\n"
                    f"Fold sizes: {[len(self.X_train) // cv_folds] * cv_folds}")

        except Exception as e:
            self.show_error(f"Error in data splitting: {str(e)}")

    def apply_dimensionality_reduction(self):
        """Apply selected dimensionality reduction method"""
        try:
            if self.X_train is None:
                raise ValueError("No data loaded. Please load a dataset first.")

            method = self.dimred_combo.currentText()
            n_components = 2  # Default for visualization

            if method == "PCA":
                n_components = self.pca_n_components.value()
                reducer = PCA(n_components=n_components)
                reduced_data = reducer.fit_transform(self.X_train)

                # Plot explained variance
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.bar(range(1, len(reducer.explained_variance_ratio_) + 1),
                       reducer.explained_variance_ratio_)
                ax.set_title("PCA Explained Variance Ratio")
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Explained Variance Ratio")
                self.canvas.draw()

            elif method == "Kernel PCA":
                n_components = self.kpca_n_components.value()
                kernel = self.kpca_kernel.currentText()
                reducer = KernelPCA(n_components=n_components, kernel=kernel)
                reduced_data = reducer.fit_transform(self.X_train)

            elif method == "LDA":
                if self.y_train is None:
                    raise ValueError("LDA requires labeled data")
                n_components = self.lda_n_components.value()
                reducer = LinearDiscriminantAnalysis(n_components=n_components)
                reduced_data = reducer.fit_transform(self.X_train, self.y_train)

                # Calculate class separation metrics
                between_class_variance = np.sum(reducer.explained_variance_ratio_)
                self.metrics_text.setText(
                    f"LDA Class Separation Metrics:\n"
                    f"Between-class variance: {between_class_variance:.4f}\n"
                    f"Explained variance ratio: {reducer.explained_variance_ratio_}")

            elif method == "t-SNE":
                n_components = self.tsne_n_components.value()
                perplexity = self.tsne_perplexity.value()
                reducer = TSNE(n_components=n_components, perplexity=perplexity)
                reduced_data = reducer.fit_transform(self.X_train)

            elif method == "MDS":
                n_components = self.mds_n_components.value()
                reducer = MDS(n_components=n_components)
                reduced_data = reducer.fit_transform(self.X_train)

            elif method == "Isomap":
                n_components = self.isomap_n_components.value()
                n_neighbors = self.isomap_n_neighbors.value()
                reducer = Isomap(n_components=n_components, n_neighbors=n_neighbors)
                reduced_data = reducer.fit_transform(self.X_train)

            else:
                raise ValueError(f"Method {method} not implemented")

            # Visualize the reduced data
            self.visualize_reduced_data(reduced_data, method)

            self.current_reducer = reducer
            self.status_bar.showMessage(f"Applied {method} dimensionality reduction")

        except Exception as e:
            self.show_error(f"Error in dimensionality reduction: {str(e)}")

    def visualize_reduced_data(self, reduced_data, method_name):
        """Visualize the reduced dimensionality data"""
        self.figure.clear()

        if reduced_data.shape[1] == 1:
            # 1D visualization - histogram
            ax = self.figure.add_subplot(111)
            ax.hist(reduced_data, bins=30)
            ax.set_title(f"{method_name} - 1D Projection")
            ax.set_xlabel("Component 1")

        elif reduced_data.shape[1] >= 2:
            if self.viz_3d_btn.isChecked() and reduced_data.shape[1] >= 3:
                # 3D visualization
                ax = self.figure.add_subplot(111, projection='3d')
                if self.y_train is not None:
                    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                                         c=self.y_train, cmap='viridis')
                    self.figure.colorbar(scatter)
                else:
                    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2])
                ax.set_title(f"{method_name} - 3D Projection")
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
            else:
                # 2D visualization
                ax = self.figure.add_subplot(111)
                if self.y_train is not None:
                    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.y_train, cmap='viridis')
                    self.figure.colorbar(scatter)
                else:
                    ax.scatter(reduced_data[:, 0], reduced_data[:, 1])
                ax.set_title(f"{method_name} - 2D Projection")
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")

        self.canvas.draw()

    def run_elbow_method(self):
        """Run elbow method to determine optimal number of clusters"""
        try:
            if self.X_train is None:
                raise ValueError("No data loaded. Please load a dataset first.")

            # Calculate inertia for different k values
            k_range = range(1, 11)
            inertias = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(self.X_train)
                inertias.append(kmeans.inertia_)

            # Plot the elbow curve
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(k_range, inertias, 'bo-')
            ax.set_title("Elbow Method for Optimal k")
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel("Inertia")
            self.canvas.draw()

            self.status_bar.showMessage("Elbow method completed. Look for the 'elbow' in the plot.")

        except Exception as e:
            self.show_error(f"Error in elbow method: {str(e)}")

    def calculate_silhouette_score(self):
        """Calculate silhouette score for clustering"""
        try:
            if self.X_train is None:
                raise ValueError("No data loaded. Please load a dataset first.")

            # Get optimal k from user (could be automated)
            k = QInputDialog.getInt(self, "Silhouette Score",
                                    "Enter number of clusters:", 2, 2, 20)[0]

            # Cluster the data
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.X_train)

            # Calculate silhouette score
            score = silhouette_score(self.X_train, cluster_labels)

            self.metrics_text.setText(
                f"Clustering Evaluation:\n"
                f"Silhouette Score for k={k}: {score:.4f}\n\n"
                f"Interpretation:\n"
                f"1.0: Perfect clustering\n"
                f"0.0: Overlapping clusters\n"
                f"-1.0: Incorrect clustering")

            self.status_bar.showMessage(f"Silhouette score for k={k}: {score:.4f}")

        except Exception as e:
            self.show_error(f"Error calculating silhouette score: {str(e)}")

    def compute_eigen_projection(self):
        """Compute eigenvectors and project data into 1D for the given covariance matrix"""
        try:
            # Given covariance matrix
            cov_matrix = np.array([[5, 2], [2, 3]])

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Sort by eigenvalues (descending)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Project data into 1D using the first eigenvector
            if self.X_train is not None and len(self.X_train.shape) == 2 and self.X_train.shape[1] == 2:
                # If we have 2D data, project it
                projection = np.dot(self.X_train, eigenvectors[:, 0])
                projected_data = projection.reshape(-1, 1)

                # Visualize the projection
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.scatter(projection, np.zeros_like(projection), c=self.y_train if self.y_train is not None else 'b')
                ax.set_title("1D Projection using Dominant Eigenvector")
                ax.set_xlabel("Projection Axis")
                self.canvas.draw()

                self.status_bar.showMessage("Data projected to 1D using dominant eigenvector")
            else:
                # Just show the eigen decomposition
                self.metrics_text.setText(
                    f"Eigen Decomposition of Covariance Matrix:\n"
                    f"Σ = {cov_matrix}\n\n"
                    f"Eigenvalues: {eigenvalues}\n"
                    f"Eigenvectors:\n{eigenvectors}\n\n"
                    f"Dominant eigenvector (for projection): {eigenvectors[:, 0]}")

                self.status_bar.showMessage("Computed eigenvectors for given covariance matrix")

        except Exception as e:
            self.show_error(f"Error in eigen projection: {str(e)}")

    # [Rest of the existing methods remain unchanged]


def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()