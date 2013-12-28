import pyqtgraph as pg
pg.mkQApp()

def test_combobox():
    cb = pg.ComboBox()
    items = {'a': 1, 'b': 2, 'c': 3}
    cb.setItems(items)
    cb.setValue(2)
    assert str(cb.currentText()) == 'b'
    assert cb.value() == 2
    
    # Clear item list; value should be None
    cb.clear()
    assert cb.value() == None
    
    # Reset item list; value should be set automatically
    cb.setItems(items)
    assert cb.value() == 2
    
    # Clear item list; repopulate with same names and new values
    items = {'a': 4, 'b': 5, 'c': 6}
    cb.clear()
    cb.setItems(items)
    assert cb.value() == 5
    
    # Set list instead of dict
    cb.setItems(items.keys())
    assert str(cb.currentText()) == 'b'
    
    cb.setValue('c')
    assert cb.value() == str(cb.currentText())
    assert cb.value() == 'c'
    
    cb.setItemValue('c', 7)
    assert cb.value() == 7
    
    
if __name__ == '__main__':
    cb = pg.ComboBox()
    cb.show()
    cb.setItems({'': None, 'a': 1, 'b': 2, 'c': 3})
    def fn(ind):
        print "New value:", cb.value()
    cb.currentIndexChanged.connect(fn)