import sys
from PyQt5 import QtCore
from PyQt5.QtCore import QRectF, QRect, QLineF, pyqtSlot
from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QVBoxLayout, \
    QApplication, QLineEdit, QGraphicsScene, QGraphicsView
from PIL import Image
import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
hidden = 54
cl = 10

class GraphicsScene(QGraphicsScene):
    def __init__(self, h, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.w = h
        self.h = h
        self.setSceneRect(0, 0, self.w, self.h)
        self.setBackgroundBrush(QtCore.Qt.black)

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setScene(self)
        self.points = []

    def mouseMoveEvent(self, event):
        pen = QPen(QtCore.Qt.white, 17.0, QtCore.Qt.SolidLine)
        brush = QBrush(QtCore.Qt.white)
        x = event.scenePos().x()
        y = event.scenePos().y()
        self.points.append((x, y))
        if len(self.points) == 1:
            self.addEllipse(self.points[0][0], self.points[0][1], 1, 1, pen, brush)
        else:
            self.addLine(QLineF(self.points[-2][0], self.points[-2][1],
                                 self.points[-1][0], self.points[-1][1]), pen)

    def mouseReleaseEvent(self, QGraphicsSceneMouseEvent):
        self.points.clear()


class Drawing(QWidget):
    def __init__(self, w, h):
        super().__init__()
        self.setGeometry(w / 4, h / 4, 359, 359)
        self.setWindowTitle('Digits recognizer')

        self.scene = GraphicsScene(280, self)

        self.recognize = QPushButton('Recognize')
        self.recognize.clicked.connect(self.save_image)
        self.clear = QPushButton('Clear')
        self.clear.clicked.connect(self.del_digit)

        self.result = QLineEdit()
        self.result.setReadOnly(True)

        self.set_layout()

        self.show()

    def set_layout(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.scene.view)
        main_layout.addWidget(self.result)

        layout = QHBoxLayout()
        layout.addWidget(self.recognize)
        layout.addWidget(self.clear)

        main_layout.addLayout(layout)
        self.setLayout(main_layout)

    @pyqtSlot()
    def save_image(self):
        outputimg = QPixmap(280, 280)
        painter = QPainter(outputimg)
        targetrect = QRectF(0, 0, 280, 280)
        sourcerect = QRect(0, 0, 280, 280)
        self.scene.view.render(painter, targetrect, sourcerect)
        outputimg.save("output.png", "PNG")
        painter.end()
        self.convert_to_grayscale()

    def convert_to_grayscale(self):
        img = Image.open("output.png").convert('L')
        img.thumbnail((28, 28), Image.ANTIALIAS)
        img.save("output.png", "PNG")
        data = list(img.getdata()) + [1]

        digit = np.array([data])
        # weights matrix in->hidden
        W1 = np.loadtxt('W1', usecols=range(hidden), delimiter=',')
        # weights matrix hidden->out, hid+"bias unit"
        W2 = np.loadtxt('W2', usecols=range(cl), delimiter=',')

        Out1 = sigmoid(digit.dot(W1))
        # ones column
        Out1 = np.append(Out1, np.ones((Out1.shape[0], 1)), 1)
        # sample output, dimOut2=examples*out
        Out2 = sigmoid(Out1.dot(W2))
        out = "Answer: {:d}. Probability {:2.1f}%".\
            format(np.argmax(Out2), np.max(Out2) * 100)
        self.result.setText(out)

    @pyqtSlot()
    def del_digit(self):
        self.scene.clear()
        self.result.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    width, height = screen_resolution.width(), screen_resolution.height()
    window = Drawing(width, height)
    sys.exit(app.exec_())
