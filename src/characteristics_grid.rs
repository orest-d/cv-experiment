const CHARACTERISTICS_BUFFER_SIZE: usize = 640 * 480;


#[derive(Debug, Clone, Copy)]
pub struct Characteristics {
    pub angle: u8,
    pub intensity: u8,
}

pub struct CharacteristicsGrid {
    pub cols: usize,
    pub rows: usize,
    pub data: [Characteristics; CHARACTERISTICS_BUFFER_SIZE],
}

impl CharacteristicsGrid {
    pub fn new(cols: usize, rows: usize) -> CharacteristicsGrid {
        let len = rows * cols;
        assert!(len < CHARACTERISTICS_BUFFER_SIZE);
        CharacteristicsGrid {
            rows: rows,
            cols: cols,
            data: [Characteristics {
                angle: 0,
                intensity: 0,
            }; CHARACTERISTICS_BUFFER_SIZE],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> Characteristics {
        self.data[x + self.cols * y]
    }

    pub fn set(&mut self, x: usize, y: usize, value: Characteristics) {
        self.data[x * self.cols + y] = value
    }
}

