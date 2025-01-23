import sys
from PyQt5 import QtWidgets
from gui.main_menu import MainApp

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()  # Corrigido para instanciar a classe correta
    window.show()
    sys.exit(app.exec_())