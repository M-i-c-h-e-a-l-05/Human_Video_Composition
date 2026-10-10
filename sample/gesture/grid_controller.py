import cv2


class GridController:

    def __init__(self, rows=6, cols=8):

        self.rows = rows
        self.cols = cols

        self.active = False

        self.selected_cells = set()

        self.selection_active = False

        self.grid_rectangle = None
        self.button_rectangle = None

        self.previous_cells = set()

    # --------------------------------------------------
    # GRID
    # --------------------------------------------------

    def update_grid(self, width, height):

        self.grid_rectangle = (
            0,
            0,
            width,
            height
        )

        # Top-center Grid button
        button_width = 100
        button_height = 60

        x1 = (width - button_width) // 2
        y1 = 20

        self.button_rectangle = (
            x1,
            y1,
            x1 + button_width,
            y1 + button_height
        )

    def activate(self, width, height):

        self.active = True

        self.selected_cells.clear()

        self.selection_active = False

        self.previous_cells.clear()

        self.update_grid(
            width,
            height
        )

    def deactivate(self):

        self.active = False

        self.selected_cells.clear()

        self.selection_active = False

        self.previous_cells.clear()

    # --------------------------------------------------
    # BUTTON
    # --------------------------------------------------

    def point_inside_button(self, x, y):

        if self.button_rectangle is None:
            return False

        x1, y1, x2, y2 = self.button_rectangle

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    # --------------------------------------------------
    # CELL
    # --------------------------------------------------

    def get_cell(self, x, y):

        if self.grid_rectangle is None:
            return None

        x1, y1, x2, y2 = self.grid_rectangle

        if not (
            x1 <= x < x2
            and
            y1 <= y < y2
        ):
            return None

        grid_width = x2 - x1
        grid_height = y2 - y1

        col = int(
            (x - x1)
            /
            grid_width
            *
            self.cols
        )

        row = int(
            (y - y1)
            /
            grid_height
            *
            self.rows
        )

        col = max(
            0,
            min(
                self.cols - 1,
                col
            )
        )

        row = max(
            0,
            min(
                self.rows - 1,
                row
            )
        )

        return row, col

    # --------------------------------------------------
    # CELL RECTANGLE
    # --------------------------------------------------

    def get_cell_rectangle(self, row, col):

        x1, y1, x2, y2 = self.grid_rectangle

        cell_width = (
            x2 - x1
        ) / self.cols

        cell_height = (
            y2 - y1
        ) / self.rows

        cell_x1 = int(
            x1 + col * cell_width
        )

        cell_y1 = int(
            y1 + row * cell_height
        )

        cell_x2 = int(
            x1 + (col + 1) * cell_width
        )

        cell_y2 = int(
            y1 + (row + 1) * cell_height
        )

        return (
            cell_x1,
            cell_y1,
            cell_x2,
            cell_y2
        )

    # --------------------------------------------------
    # SELECTION
    # --------------------------------------------------

    def start_selection(
        self,
        index_position,
        middle_position
    ):

        if not self.active:
            return

        self.selection_active = True

        self.previous_cells.clear()

        self.add_finger_cells(
            index_position,
            middle_position
        )

    def update_selection(
        self,
        index_position,
        middle_position
    ):

        if not self.active:
            return

        if not self.selection_active:
            self.start_selection(
                index_position,
                middle_position
            )

            return

        self.add_finger_cells(
            index_position,
            middle_position
        )

    def add_finger_cells(
        self,
        index_position,
        middle_position
    ):

        index_cell = self.get_cell(
            *index_position
        )

        middle_cell = self.get_cell(
            *middle_position
        )

        current_cells = set()

        if index_cell is not None:
            current_cells.add(index_cell)

        if middle_cell is not None:
            current_cells.add(middle_cell)

        # Add cells crossed between
        # previous and current finger positions
        for previous_cell in self.previous_cells:

            for current_cell in current_cells:

                crossed = self.get_cells_between(
                    previous_cell,
                    current_cell
                )

                self.selected_cells.update(
                    crossed
                )

        # Add current cells
        self.selected_cells.update(
            current_cells
        )

        self.previous_cells = current_cells

    # --------------------------------------------------
    # PATH BETWEEN CELLS
    # --------------------------------------------------

    def get_cells_between(
        self,
        cell1,
        cell2
    ):

        row1, col1 = cell1
        row2, col2 = cell2

        cells = set()

        row_step = (
            1
            if row2 >= row1
            else -1
        )

        col_step = (
            1
            if col2 >= col1
            else -1
        )

        row = row1

        while True:

            col = col1

            while True:

                cells.add(
                    (row, col)
                )

                if col == col2:
                    break

                col += col_step

            if row == row2:
                break

            row += row_step

        return cells

    # --------------------------------------------------
    # FINISH
    # --------------------------------------------------

    def finish_selection(self):

        self.selection_active = False

        self.previous_cells.clear()

        if not self.selected_cells:
            return None

        return self.get_selected_rectangle()

    # --------------------------------------------------
    # SELECTED RECTANGLE
    # --------------------------------------------------

    def get_selected_rectangle(self):

        if not self.selected_cells:
            return None

        rows = [
            cell[0]
            for cell in self.selected_cells
        ]

        cols = [
            cell[1]
            for cell in self.selected_cells
        ]

        min_row = min(rows)
        max_row = max(rows)

        min_col = min(cols)
        max_col = max(cols)

        first = self.get_cell_rectangle(
            min_row,
            min_col
        )

        last = self.get_cell_rectangle(
            max_row,
            max_col
        )

        return (
            first[0],
            first[1],
            last[2],
            last[3]
        )

    # --------------------------------------------------
    # CREATE CUT
    # --------------------------------------------------

    def create_cut(self, cut_controller):

        rectangle = self.get_selected_rectangle()

        if rectangle is None:
            return False

        cut_controller.rectangle = rectangle

        cut_controller.cut_source_rectangle = rectangle

        cut_controller.cut_active = True

        cut_controller.mode = "SELECTED"

        return True

    # --------------------------------------------------
    # DRAW GRID
    # --------------------------------------------------

    def draw(self, frame):

        if not self.active:
            return frame

        if self.grid_rectangle is None:
            return frame

        x1, y1, x2, y2 = (
            self.grid_rectangle
        )

        # Draw selected cells
        for row, col in self.selected_cells:

            cx1, cy1, cx2, cy2 = (
                self.get_cell_rectangle(
                    row,
                    col
                )
            )

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (cx1, cy1),
                (cx2, cy2),
                (255, 255, 255),
                -1
            )

            frame[:] = cv2.addWeighted(
                frame,
                0.25,
                overlay,
                0.75,
                0
            )

        # Draw vertical lines
        cell_width = (
            x2 - x1
        ) / self.cols

        cell_height = (
            y2 - y1
        ) / self.rows

        for col in range(
            self.cols + 1
        ):

            x = int(
                x1 + col * cell_width
            )

            cv2.line(
                frame,
                (x, y1),
                (x, y2),
                (180, 180, 180),
                1
            )

        # Draw horizontal lines
        for row in range(
            self.rows + 1
        ):

            y = int(
                y1 + row * cell_height
            )

            cv2.line(
                frame,
                (x1, y),
                (x2, y),
                (180, 180, 180),
                1
            )

        return frame