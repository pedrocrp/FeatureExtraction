import sys
from PyQt5 import QtWidgets
from gui.gui_interface import FeatureExtractionApp

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = FeatureExtractionApp()
    window.show()
    sys.exit(app.exec_())
