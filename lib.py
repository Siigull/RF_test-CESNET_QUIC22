from sklearn.neighbors import KNeighborsClassifier

class knn():
    def initialize(self, X, y):
        self.X = X 
        self.y = y

    def fit(self):
        clf = KNeighborsClassifier()
        clf.fit(self.X, self.y)
    
    def predict_proba(self, class_index):
