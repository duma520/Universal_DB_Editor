# -*- coding: utf-8 -*-
import sys
import os
import sqlite3
import logging
import csv
import json
import shutil
import time
import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, 
    QFileDialog, QMessageBox, QInputDialog, QHeaderView, 
    QMenu, QAction, QSizePolicy, QSplitter, QLineEdit, 
    QComboBox, QFormLayout, QGroupBox, QProgressDialog, 
    QCheckBox, QTabWidget, QDialog, QDialogButtonBox,
    QSpinBox, QSlider
)
from PyQt5.QtCore import (
    Qt, pyqtSignal, QThread, QTimer, QSize, 
    QDateTime, QObject, QMutex, QWaitCondition
)
from PyQt5.QtGui import (
    QCursor, QPixmap, QImage, QIcon, 
    QPainter, QColor, QFont
)

# 配置日志记录
logging.basicConfig(
    filename='universal_db_editor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# 全局互斥锁用于线程安全
global_mutex = QMutex()

class DatabaseSchema:
    """数据库模式分析器"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.tables = []
        self.primary_keys = {}
        self.column_types = {}
        self.analyze_schema()

    def analyze_schema(self) -> None:
        """分析数据库结构"""
        try:
            conn = self._create_connection()
            if not conn:
                return

            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            self.tables = [row[0] for row in cursor.fetchall() if row[0] != 'sqlite_sequence']

            # 分析每个表的结构
            for table in self.tables:
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # 提取主键
                self.primary_keys[table] = [
                    col[1] for col in columns if col[5] == 1
                ]
                
                # 存储列类型
                self.column_types[table] = {
                    col[1]: col[2] for col in columns
                }

            conn.close()
        except Exception as e:
            logger.error(f"分析数据库结构失败: {str(e)}", exc_info=True)
            raise

    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """创建数据库连接"""
        try:
            return sqlite3.connect(self.db_path)
        except Exception as e:
            logger.error(f"连接数据库失败: {str(e)}", exc_info=True)
            return None

    def get_display_columns(self, table: str) -> List[str]:
        """获取适合显示的列"""
        if table not in self.column_types:
            return []
        
        # 排除BLOB等不适合直接显示的列
        return [
            col for col, dtype in self.column_types[table].items()
            if not dtype.upper().startswith('BLOB')
        ]

class DatabaseAdapter(QObject):
    """通用数据库适配器"""
    operation_complete = pyqtSignal(bool, str)
    progress_updated = pyqtSignal(int, int, str)

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.schema = DatabaseSchema(db_path)
        self.current_table = self.schema.tables[0] if self.schema.tables else None
        self.connection_pool = []
        self.max_connections = 5
        self._init_connection_pool()

    def _init_connection_pool(self) -> None:
        """初始化连接池"""
        try:
            for _ in range(self.max_connections):
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=10,
                    isolation_level=None,  # 自动提交模式
                    check_same_thread=False  # 允许多线程访问
                )
                conn.execute("PRAGMA journal_mode=WAL")  # 提高并发性能
                self.connection_pool.append(conn)
        except Exception as e:
            logger.error(f"初始化连接池失败: {str(e)}", exc_info=True)
            self._close_all_connections()
            raise

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """从连接池获取连接"""
        global_mutex.lock()
        try:
            if not self.connection_pool:
                # 如果没有可用连接，等待或创建新连接
                if len(self.connection_pool) < self.max_connections * 2:  # 允许临时扩展
                    conn = sqlite3.connect(
                        self.db_path,
                        timeout=10,
                        isolation_level=None,
                        check_same_thread=False
                    )
                    conn.execute("PRAGMA journal_mode=WAL")
                    return conn
                return None
            return self.connection_pool.pop()
        finally:
            global_mutex.unlock()

    def _release_connection(self, conn: sqlite3.Connection) -> None:
        """释放连接到连接池"""
        global_mutex.lock()
        try:
            if len(self.connection_pool) < self.max_connections:
                self.connection_pool.append(conn)
            else:
                conn.close()
        finally:
            global_mutex.unlock()

    def _close_all_connections(self) -> None:
        """关闭所有连接"""
        global_mutex.lock()
        try:
            for conn in self.connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self.connection_pool = []
        finally:
            global_mutex.unlock()

    def __del__(self):
        self._close_all_connections()

    def execute_query(
        self,
        query: str,
        params: Tuple = (),
        fetch_all: bool = True,
        commit: bool = False
    ) -> Union[List[Tuple], None]:
        """执行SQL查询"""
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            if not conn:
                raise Exception("无法获取数据库连接")

            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if commit:
                conn.commit()
            
            if fetch_all:
                return cursor.fetchall()
            return None

        except sqlite3.Error as e:
            logger.error(f"执行查询失败: {query} | 参数: {params} | 错误: {str(e)}")
            if conn and commit:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._release_connection(conn)

    def get_table_data(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        params: Tuple = (),
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """获取表数据"""
        if table not in self.schema.tables:
            raise ValueError(f"表 {table} 不存在")

        if not columns:
            columns = self.schema.get_display_columns(table)
            if not columns:
                columns = ['*']

        query = f"SELECT {', '.join(columns)} FROM {table}"
        if where:
            query += f" WHERE {where}"
        if limit > 0:
            query += f" LIMIT {limit}"

        try:
            rows = self.execute_query(query, params)
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"获取表数据失败: {str(e)}", exc_info=True)
            raise

    def update_record(
        self,
        table: str,
        record_id: Any,
        data: Dict[str, Any],
        id_column: Optional[str] = None
    ) -> bool:
        """更新记录"""
        if table not in self.schema.tables:
            raise ValueError(f"表 {table} 不存在")

        if not id_column:
            id_column = self.schema.primary_keys[table][0] if self.schema.primary_keys[table] else 'rowid'

        set_clause = ", ".join([f"{k}=?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {id_column}=?"

        try:
            params = tuple(data.values()) + (record_id,)
            self.execute_query(query, params, fetch_all=False, commit=True)
            return True
        except Exception as e:
            logger.error(f"更新记录失败: {str(e)}", exc_info=True)
            return False

    def insert_record(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> Optional[Any]:
        """插入新记录"""
        if table not in self.schema.tables:
            raise ValueError(f"表 {table} 不存在")

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        try:
            self.execute_query(query, tuple(data.values()), fetch_all=False, commit=True)
            return self.execute_query("SELECT last_insert_rowid()")[0][0]
        except Exception as e:
            logger.error(f"插入记录失败: {str(e)}", exc_info=True)
            return None

    def delete_record(
        self,
        table: str,
        record_id: Any,
        id_column: Optional[str] = None
    ) -> bool:
        """删除记录"""
        if table not in self.schema.tables:
            raise ValueError(f"表 {table} 不存在")

        if not id_column:
            id_column = self.schema.primary_keys[table][0] if self.schema.primary_keys[table] else 'rowid'

        query = f"DELETE FROM {table} WHERE {id_column}=?"

        try:
            self.execute_query(query, (record_id,), fetch_all=False, commit=True)
            return True
        except Exception as e:
            logger.error(f"删除记录失败: {str(e)}", exc_info=True)
            return False

    def search_records(
        self,
        table: str,
        search_column: str,
        search_value: str,
        exact_match: bool = False
    ) -> List[Dict[str, Any]]:
        """搜索记录"""
        if table not in self.schema.tables:
            raise ValueError(f"表 {table} 不存在")

        if search_column not in self.schema.column_types[table]:
            raise ValueError(f"列 {search_column} 不存在")

        # 根据列类型决定搜索方式
        column_type = self.schema.column_types[table][search_column].upper()
        
        if exact_match or column_type in ('INTEGER', 'REAL', 'NUMERIC'):
            where = f"{search_column} = ?"
            params = (search_value,)
        else:
            where = f"{search_column} LIKE ?"
            params = (f"%{search_value}%",)

        return self.get_table_data(table, where=where, params=params)

class UniversalDBEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("通用SQLite数据库编辑器 v4.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # 数据库相关
        self.db_path = None
        self.db_adapter = None
        self.current_table = None
        
        # UI组件
        self.table_widget = None
        self.table_selector = None
        self.search_input = None
        self.search_column = None
        
        # 初始化UI
        self.init_ui()
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
        
        # 连接信号
        self.connect_signals()
        
        # 加载最近使用的数据库
        self.load_last_db()

    def init_ui(self):
        """初始化用户界面"""
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 顶部工具栏
        toolbar = self.create_toolbar()
        main_layout.addWidget(toolbar)
        
        # 主内容区域
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧表格区域
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 表选择器
        self.table_selector = QComboBox()
        self.table_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        left_layout.addWidget(self.table_selector)
        
        # 表格控件
        self.table_widget = SortableTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_widget.setSelectionMode(QTableWidget.ExtendedSelection)
        left_layout.addWidget(self.table_widget)
        
        # 右侧详情区域
        right_panel = QTabWidget()
        
        # 详情标签页
        detail_tab = QWidget()
        detail_layout = QVBoxLayout(detail_tab)
        
        self.detail_text = QLabel("选择记录查看详情")
        self.detail_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.detail_text.setWordWrap(True)
        self.detail_text.setStyleSheet("font-family: monospace;")
        
        detail_layout.addWidget(self.detail_text)
        right_panel.addTab(detail_tab, "记录详情")
        
        # 结构标签页
        schema_tab = QWidget()
        schema_layout = QVBoxLayout(schema_tab)
        
        self.schema_text = QLabel("数据库结构信息")
        self.schema_text.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.schema_text.setWordWrap(True)
        self.schema_text.setStyleSheet("font-family: monospace;")
        
        schema_layout.addWidget(self.schema_text)
        right_panel.addTab(schema_tab, "数据库结构")
        
        # 将左右面板添加到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        self.setCentralWidget(main_widget)
        
        # 初始化右键菜单
        self.init_context_menu()

    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 数据库操作按钮
        self.btn_open = QPushButton("打开数据库")
        self.btn_open.setToolTip("打开SQLite数据库文件")
        
        self.btn_save = QPushButton("保存更改")
        self.btn_save.setToolTip("保存所有更改到数据库")
        self.btn_save.setEnabled(False)
        
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setToolTip("刷新当前表数据")
        
        # 记录操作按钮
        self.btn_add = QPushButton("添加记录")
        self.btn_add.setToolTip("添加新记录")
        
        self.btn_edit = QPushButton("编辑记录")
        self.btn_edit.setToolTip("编辑选中记录")
        self.btn_edit.setEnabled(False)
        
        self.btn_delete = QPushButton("删除记录")
        self.btn_delete.setToolTip("删除选中记录")
        self.btn_delete.setEnabled(False)
        
        # 搜索区域
        search_group = QGroupBox("搜索")
        search_layout = QHBoxLayout(search_group)
        
        self.search_column = QComboBox()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入搜索内容...")
        self.btn_search = QPushButton("搜索")
        
        search_layout.addWidget(self.search_column)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.btn_search)
        
        # 添加到工具栏
        layout.addWidget(self.btn_open)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_refresh)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_edit)
        layout.addWidget(self.btn_delete)
        layout.addWidget(search_group)
        
        return toolbar

    def connect_signals(self):
        """连接信号与槽"""
        self.btn_open.clicked.connect(self.open_database)
        self.btn_save.clicked.connect(self.save_changes)
        self.btn_refresh.clicked.connect(self.refresh_data)
        self.btn_add.clicked.connect(self.add_record)
        self.btn_edit.clicked.connect(self.edit_record)
        self.btn_delete.clicked.connect(self.delete_record)
        self.btn_search.clicked.connect(self.perform_search)
        
        self.table_selector.currentTextChanged.connect(self.table_selected)
        self.table_widget.itemSelectionChanged.connect(self.update_record_detail)
        self.table_widget.doubleClicked.connect(self.edit_record)
        
        self.search_input.returnPressed.connect(self.perform_search)

    def init_context_menu(self):
        """初始化右键菜单"""
        self.context_menu = QMenu(self)
        
        # 添加操作
        actions = [
            ("编辑记录", self.edit_record),
            ("添加记录", self.add_record),
            ("删除记录", self.delete_record),
            None,  # 分隔线
            ("复制值", self.copy_cell_value),
            ("复制行", self.copy_row_data),
            None,
            ("导出选中行", self.export_selected),
            ("搜索选中内容", self.search_selected)
        ]
        
        for action in actions:
            if action is None:
                self.context_menu.addSeparator()
            else:
                text, handler = action
                act = QAction(text, self)
                act.triggered.connect(handler)
                self.context_menu.addAction(act)
        
        self.table_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_widget.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos):
        """显示右键菜单"""
        if self.table_widget.selectionModel().hasSelection():
            self.context_menu.exec_(self.table_widget.viewport().mapToGlobal(pos))

    def load_last_db(self):
        """加载上次使用的数据库"""
        settings = QApplication.instance().settings if hasattr(QApplication.instance(), 'settings') else {}
        last_db = settings.get('last_db_path', None)
        
        if last_db and os.path.exists(last_db):
            self.open_database(last_db)

    def open_database(self, db_path=None):
        """打开数据库文件"""
        try:
            if not db_path:
                db_path, _ = QFileDialog.getOpenFileName(
                    self, "打开SQLite数据库", "", "SQLite数据库 (*.db *.sqlite *.sqlite3)")
                
                if not db_path:
                    return
            
            # 验证文件
            if not os.path.exists(db_path):
                QMessageBox.warning(self, "错误", "数据库文件不存在")
                return
                
            if not os.access(db_path, os.R_OK):
                QMessageBox.warning(self, "错误", "无法读取数据库文件")
                return
                
            # 尝试连接数据库
            try:
                self.db_adapter = DatabaseAdapter(db_path)
                self.db_path = db_path
                
                # 更新UI
                self.update_table_list()
                self.update_schema_info()
                
                # 保存最近使用的数据库
                if hasattr(QApplication.instance(), 'settings'):
                    QApplication.instance().settings['last_db_path'] = db_path
                
                self.status_bar.showMessage(f"已加载数据库: {os.path.basename(db_path)}")
                logger.info(f"成功打开数据库: {db_path}")
                
            except Exception as e:
                logger.error(f"打开数据库失败: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "错误", f"打开数据库失败: {str(e)}")
                self.status_bar.showMessage("打开数据库失败")
                
        except Exception as e:
            logger.error(f"打开数据库出错: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"打开数据库时发生错误: {str(e)}")

    def update_table_list(self):
        """更新表列表"""
        if not self.db_adapter:
            return
            
        self.table_selector.clear()
        self.table_selector.addItems(self.db_adapter.schema.tables)
        
        if self.db_adapter.schema.tables:
            self.current_table = self.db_adapter.schema.tables[0]
            self.load_table_data(self.current_table)

    def table_selected(self, table_name):
        """表选择变化"""
        if table_name and table_name != self.current_table:
            self.current_table = table_name
            self.load_table_data(table_name)

    def load_table_data(self, table_name, where=None, params=()):
        """加载表数据"""
        try:
            if not self.db_adapter or table_name not in self.db_adapter.schema.tables:
                return
                
            # 显示加载状态
            self.status_bar.showMessage(f"正在加载表 {table_name}...")
            QApplication.processEvents()
            
            # 获取表数据
            columns = self.db_adapter.schema.get_display_columns(table_name)
            data = self.db_adapter.get_table_data(table_name, columns, where, params)
            
            if not data:
                self.table_widget.clear()
                self.table_widget.setRowCount(0)
                self.table_widget.setColumnCount(0)
                self.status_bar.showMessage(f"表 {table_name} 为空")
                return
                
            # 配置表格
            self.table_widget.setColumnCount(len(columns))
            self.table_widget.setHorizontalHeaderLabels(columns)
            self.table_widget.setRowCount(len(data))
            
            # 填充数据
            for row_idx, row_data in enumerate(data):
                for col_idx, col_name in enumerate(columns):
                    value = row_data.get(col_name, "")
                    item = QTableWidgetItem(str(value) if value is not None else "NULL")
                    
                    # 根据数据类型设置对齐方式
                    col_type = self.db_adapter.schema.column_types[table_name].get(col_name, "").upper()
                    if 'INT' in col_type or 'REAL' in col_type or 'NUM' in col_type:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    elif 'TEXT' in col_type or 'CHAR' in col_type:
                        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    else:
                        item.setTextAlignment(Qt.AlignCenter)
                    
                    self.table_widget.setItem(row_idx, col_idx, item)
            
            # 调整列宽
            self.table_widget.resizeColumnsToContents()
            
            # 更新状态
            self.status_bar.showMessage(f"已加载表 {table_name}, 共 {len(data)} 条记录")
            
            # 更新搜索列选择
            self.update_search_columns(table_name)
            
        except Exception as e:
            logger.error(f"加载表数据失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"加载表数据失败: {str(e)}")
            self.status_bar.showMessage("加载数据失败")

    def update_search_columns(self, table_name):
        """更新可搜索的列"""
        if not self.db_adapter or table_name not in self.db_adapter.schema.tables:
            return
            
        self.search_column.clear()
        self.search_column.addItems(self.db_adapter.schema.get_display_columns(table_name))

    def update_record_detail(self):
        """更新记录详情"""
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            self.detail_text.setText("未选择记录")
            return
            
        # 获取选中行的所有数据
        row = selected_items[0].row()
        columns = [self.table_widget.horizontalHeaderItem(i).text() for i in range(self.table_widget.columnCount())]
        
        detail_text = []
        for col_idx, col_name in enumerate(columns):
            item = self.table_widget.item(row, col_idx)
            value = item.text() if item else "NULL"
            detail_text.append(f"{col_name}: {value}")
        
        self.detail_text.setText("\n".join(detail_text))

    def update_schema_info(self):
        """更新数据库结构信息"""
        if not self.db_adapter:
            return
            
        schema_info = []
        for table in self.db_adapter.schema.tables:
            schema_info.append(f"表: {table}")
            schema_info.append("主键: " + ", ".join(self.db_adapter.schema.primary_keys[table]))
            
            for col, col_type in self.db_adapter.schema.column_types[table].items():
                schema_info.append(f"  {col}: {col_type}")
            
            schema_info.append("")
        
        self.schema_text.setText("\n".join(schema_info))

    def perform_search(self):
        """执行搜索"""
        search_text = self.search_input.text().strip()
        if not search_text or not self.current_table:
            return
            
        search_col = self.search_column.currentText()
        if not search_col:
            return
            
        try:
            # 检查是否精确匹配
            exact_match = False
            col_type = self.db_adapter.schema.column_types[self.current_table][search_col].upper()
            if col_type in ('INTEGER', 'REAL', 'NUMERIC') and search_text.isdigit():
                exact_match = True
                
            # 执行搜索
            results = self.db_adapter.search_records(
                self.current_table,
                search_col,
                search_text,
                exact_match
            )
            
            if not results:
                QMessageBox.information(self, "搜索", "未找到匹配记录")
                return
                
            # 显示结果
            self.display_search_results(results)
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"搜索失败: {str(e)}")

    def display_search_results(self, results):
        """显示搜索结果"""
        if not results or not self.current_table:
            return
            
        columns = list(results[0].keys())
        
        self.table_widget.setColumnCount(len(columns))
        self.table_widget.setHorizontalHeaderLabels(columns)
        self.table_widget.setRowCount(len(results))
        
        for row_idx, row_data in enumerate(results):
            for col_idx, col_name in enumerate(columns):
                value = row_data.get(col_name, "")
                item = QTableWidgetItem(str(value) if value is not None else "NULL")
                self.table_widget.setItem(row_idx, col_idx, item)
        
        self.table_widget.resizeColumnsToContents()
        self.status_bar.showMessage(f"找到 {len(results)} 条匹配记录")

    def add_record(self):
        """添加新记录"""
        if not self.current_table:
            return
            
        # 创建编辑对话框
        dialog = RecordEditDialog(
            self, 
            self.db_adapter.schema, 
            self.current_table,
            mode='add'
        )
        
        if dialog.exec_() == QDialog.Accepted:
            new_data = dialog.get_record_data()
            
            try:
                # 插入新记录
                new_id = self.db_adapter.insert_record(self.current_table, new_data)
                
                if new_id is not None:
                    self.status_bar.showMessage(f"成功添加记录, ID: {new_id}")
                    self.load_table_data(self.current_table)
                else:
                    QMessageBox.warning(self, "错误", "添加记录失败")
                    
            except Exception as e:
                logger.error(f"添加记录失败: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "错误", f"添加记录失败: {str(e)}")

    def edit_record(self):
        """编辑记录"""
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一条记录")
            return
            
        # 获取选中行的数据
        row = selected_items[0].row()
        columns = [self.table_widget.horizontalHeaderItem(i).text() for i in range(self.table_widget.columnCount())]
        
        record_data = {}
        for col_idx, col_name in enumerate(columns):
            item = self.table_widget.item(row, col_idx)
            record_data[col_name] = item.text() if item else ""
        
        # 获取主键值
        pk_col = self.db_adapter.schema.primary_keys[self.current_table][0] if self.db_adapter.schema.primary_keys[self.current_table] else columns[0]
        record_id = record_data.get(pk_col, None)
        
        if not record_id:
            QMessageBox.warning(self, "警告", "无法确定记录ID")
            return
            
        # 创建编辑对话框
        dialog = RecordEditDialog(
            self, 
            self.db_adapter.schema, 
            self.current_table,
            mode='edit',
            initial_data=record_data
        )
        
        if dialog.exec_() == QDialog.Accepted:
            updated_data = dialog.get_record_data()
            
            try:
                # 更新记录
                success = self.db_adapter.update_record(
                    self.current_table,
                    record_id,
                    updated_data,
                    pk_col
                )
                
                if success:
                    self.status_bar.showMessage("记录更新成功")
                    self.load_table_data(self.current_table)
                else:
                    QMessageBox.warning(self, "错误", "更新记录失败")
                    
            except Exception as e:
                logger.error(f"更新记录失败: {str(e)}", exc_info=True)
                QMessageBox.critical(self, "错误", f"更新记录失败: {str(e)}")

    def delete_record(self):
        """删除记录"""
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一条记录")
            return
            
        # 获取选中行的主键值
        row = selected_items[0].row()
        columns = [self.table_widget.horizontalHeaderItem(i).text() for i in range(self.table_widget.columnCount())]
        
        pk_col = self.db_adapter.schema.primary_keys[self.current_table][0] if self.db_adapter.schema.primary_keys[self.current_table] else columns[0]
        pk_item = self.table_widget.item(row, columns.index(pk_col)) if pk_col in columns else None
        
        if not pk_item:
            QMessageBox.warning(self, "警告", "无法确定记录ID")
            return
            
        record_id = pk_item.text()
        
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除这条记录吗? (ID: {record_id})\n此操作不可撤销!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
            
        try:
            # 删除记录
            success = self.db_adapter.delete_record(
                self.current_table,
                record_id,
                pk_col
            )
            
            if success:
                self.status_bar.showMessage("记录删除成功")
                self.load_table_data(self.current_table)
            else:
                QMessageBox.warning(self, "错误", "删除记录失败")
                
        except Exception as e:
            logger.error(f"删除记录失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"删除记录失败: {str(e)}")

    def save_changes(self):
        """保存所有更改"""
        # 在这个实现中，我们使用自动提交模式，所以此方法主要是为了UI一致性
        self.status_bar.showMessage("所有更改已自动保存")

    def refresh_data(self):
        """刷新数据"""
        if self.current_table:
            self.load_table_data(self.current_table)

    def copy_cell_value(self):
        """复制单元格值"""
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            return
            
        clipboard = QApplication.clipboard()
        clipboard.setText(selected_items[0].text())
        self.status_bar.showMessage("已复制单元格内容")

    def copy_row_data(self):
        """复制整行数据"""
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        columns = [self.table_widget.horizontalHeaderItem(i).text() for i in range(self.table_widget.columnCount())]
        
        row_data = []
        for col_idx in range(len(columns)):
            item = self.table_widget.item(row, col_idx)
            row_data.append(item.text() if item else "")
        
        clipboard = QApplication.clipboard()
        clipboard.setText("\t".join(row_data))
        self.status_bar.showMessage("已复制整行数据")

    def export_selected(self):
        """导出选中行"""
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要导出的记录")
            return
            
        # 获取所有选中行
        selected_rows = set(item.row() for item in selected_items)
        
        # 准备数据
        columns = [self.table_widget.horizontalHeaderItem(i).text() for i in range(self.table_widget.columnCount())]
        
        data = []
        for row in selected_rows:
            row_data = {}
            for col_idx, col_name in enumerate(columns):
                item = self.table_widget.item(row, col_idx)
                row_data[col_name] = item.text() if item else ""
            data.append(row_data)
        
        # 选择导出格式
        options = QFileDialog.Options()
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, "导出数据", "", 
            "CSV文件 (*.csv);;JSON文件 (*.json)", 
            options=options
        )
        
        if not file_name:
            return
            
        # 根据选择添加扩展名
        if selected_filter == "CSV文件 (*.csv)" and not file_name.endswith('.csv'):
            file_name += '.csv'
        elif selected_filter == "JSON文件 (*.json)" and not file_name.endswith('.json'):
            file_name += '.json'
        
        try:
            # 导出数据
            if selected_filter == "CSV文件 (*.csv)":
                with open(file_name, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(data)
            else:  # JSON
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.status_bar.showMessage(f"已导出 {len(selected_rows)} 条记录到 {file_name}")
            QMessageBox.information(self, "成功", f"已成功导出 {len(selected_rows)} 条记录")
            
        except Exception as e:
            logger.error(f"导出数据失败: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def search_selected(self):
        """搜索选中内容"""
        selected_items = self.table_widget.selectedItems()
        if not selected_items:
            return
            
        # 使用选中单元格的内容作为搜索条件
        selected_text = selected_items[0].text()
        selected_column = self.table_widget.horizontalHeaderItem(selected_items[0].column()).text()
        
        self.search_column.setCurrentText(selected_column)
        self.search_input.setText(selected_text)
        self.perform_search()

class RecordEditDialog(QDialog):
    """记录编辑对话框"""
    def __init__(self, parent, schema, table_name, mode='add', initial_data=None):
        super().__init__(parent)
        self.schema = schema
        self.table_name = table_name
        self.mode = mode
        self.initial_data = initial_data or {}
        
        self.setWindowTitle("添加记录" if mode == 'add' else "编辑记录")
        self.setMinimumWidth(400)
        
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout(self)
        
        self.fields = {}
        
        # 为表中的每一列创建输入控件
        for col_name, col_type in self.schema.column_types[self.table_name].items():
            # 跳过主键（如果是添加模式）
            if self.mode == 'add' and col_name in self.schema.primary_keys[self.table_name]:
                continue
                
            # 根据列类型创建适当的输入控件
            col_type = col_type.upper()
            
            if 'INT' in col_type or 'REAL' in col_type or 'NUM' in col_type:
                # 数值类型
                input_widget = QLineEdit()
                input_widget.setValidator(QDoubleValidator() if 'REAL' in col_type else QIntValidator())
            elif 'TEXT' in col_type or 'CHAR' in col_type:
                # 文本类型
                input_widget = QLineEdit()
            elif 'DATE' in col_type or 'TIME' in col_type:
                # 日期时间类型
                input_widget = QDateTimeEdit()
                input_widget.setCalendarPopup(True)
                input_widget.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            else:
                # 默认使用文本输入
                input_widget = QLineEdit()
            
            # 设置初始值
            if col_name in self.initial_data:
                if isinstance(input_widget, QDateTimeEdit):
                    try:
                        dt = QDateTime.fromString(self.initial_data[col_name], Qt.ISODate)
                        if dt.isValid():
                            input_widget.setDateTime(dt)
                    except:
                        pass
                else:
                    input_widget.setText(self.initial_data[col_name])
            
            # 添加到布局
            layout.addRow(f"{col_name} ({col_type}):", input_widget)
            self.fields[col_name] = input_widget
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        layout.addRow(button_box)
    
    def get_record_data(self):
        """获取用户输入的数据"""
        data = {}
        
        for col_name, widget in self.fields.items():
            if isinstance(widget, QLineEdit):
                value = widget.text()
            elif isinstance(widget, QDateTimeEdit):
                value = widget.dateTime().toString(Qt.ISODate)
            else:
                value = ""
                
            # 空字符串转为None
            data[col_name] = value if value else None
            
        return data

class SortableTableWidget(QTableWidget):
    """可排序的表格控件"""
    headerClicked = pyqtSignal(int)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sort_column = 0
        self._sort_order = Qt.AscendingOrder
        self.horizontalHeader().sectionClicked.connect(self.on_header_clicked)
    
    def on_header_clicked(self, column):
        """表头点击事件处理"""
        if column == self._sort_column:
            self._sort_order = Qt.DescendingOrder if self._sort_order == Qt.AscendingOrder else Qt.AscendingOrder
        else:
            self._sort_column = column
            self._sort_order = Qt.AscendingOrder
        
        self.sortItems(column, self._sort_order)
        self.headerClicked.emit(column)

if __name__ == "__main__":
    try:
        # 初始化应用
        app = QApplication(sys.argv)
        
        # 创建和应用设置
        if not hasattr(app, 'settings'):
            app.settings = {}
        
        # 创建主窗口
        window = UniversalDBEditor()
        window.show()
        
        # 运行应用
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"应用程序崩溃: {str(e)}", exc_info=True)
        QMessageBox.critical(
            None, "致命错误", 
            f"应用程序发生致命错误:\n{str(e)}\n请查看日志文件获取更多信息。"
        )