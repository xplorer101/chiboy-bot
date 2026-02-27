#!/usr/bin/env python3
"""
CHIBOY BOT - Desktop GUI Application
=========================================
A standalone desktop application with native GUI for CHIBOY trading.
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QLineEdit, QComboBox, QGroupBox, QStatusBar, QMenuBar,
    QMenu, QSplitter, QFrame, QProgressBar, QCheckBox, QSpinBox,
    QDoubleSpinBox, QScrollArea, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QAction, QIcon, QFont, QColor, QPalette
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BOT_CONFIG, TRADING_MODE, OANDA_CONFIG, BINANCE_CONFIG
from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.signals.trading_signals import TradingSignals


# ==================== ANALYSIS THREAD ====================
class AnalysisThread(QThread):
    """Background thread for market analysis."""
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def run(self):
        try:
            analyzer = MultiTimeframeAnalyzer()
            self.progress.emit("Scanning markets...")
            opportunities = analyzer.get_top_opportunities(limit=10)
            self.finished.emit(opportunities)
        except Exception as e:
            self.error.emit(str(e))


# ==================== MAIN WINDOW ====================
class CHIBOYTradingBotGUI(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = MultiTimeframeAnalyzer()
        self.signals = TradingSignals()
        self.analysis_thread = None
        self.opportunities = []
        
        self.init_ui()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging to text widget."""
        # Will be connected to the log viewer
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(f"CHIBOY BOT v{BOT_CONFIG['version']}")
        self.setMinimumSize(1200, 800)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar and main content (container)
        toolbar = self.create_toolbar()
        
        # Use a main container for toolbar + content
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add toolbar at top
        container_layout.addWidget(toolbar)
        
        # Create main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Opportunities
        left_panel = self.create_opportunities_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Details & Controls
        right_panel = self.create_details_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([700, 500])
        
        container_layout.addWidget(splitter)
        
        # Set as central widget
        self.setCentralWidget(container)
        
        # Status bar
        self.create_status_bar()
        
    def set_dark_theme(self):
        """Apply dark theme to the application."""
        dark_style = """
        QMainWindow, QWidget {
            background-color: #1e1e1e;
            color: #d4d4d4;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:disabled {
            background-color: #424242;
            color: #757575;
        }
        QPushButton.buy-btn {
            background-color: #2e7d32;
        }
        QPushButton.buy-btn:hover {
            background-color: #388e3c;
        }
        QPushButton.sell-btn {
            background-color: #c62828;
        }
        QPushButton.sell-btn:hover {
            background-color: #d32f2f;
        }
        QLabel {
            color: #d4d4d4;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QTableWidget {
            background-color: #252526;
            alternate-background-color: #2d2d30;
            gridline-color: #3c3c3c;
            border: none;
        }
        QTableWidget::item {
            padding: 8px;
        }
        QTableWidget::item:selected {
            background-color: #094771;
        }
        QHeaderView::section {
            background-color: #323233;
            color: #d4d4d4;
            padding: 6px;
            border: none;
            font-weight: bold;
        }
        QTextEdit, QLineEdit {
            background-color: #252526;
            color: #d4d4d4;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #2d2d30;
            color: #d4d4d4;
            padding: 8px 16px;
            border: none;
        }
        QTabBar::tab:selected {
            background-color: #0d47a1;
        }
        QProgressBar {
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            text-align: center;
            background-color: #252526;
        }
        QProgressBar::chunk {
            background-color: #0d47a1;
        }
        QStatusBar {
            background-color: #007acc;
            color: white;
        }
        QComboBox {
            background-color: #252526;
            color: #d4d4d4;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 4px;
        }
        QComboBox::drop-down {
            border: none;
        }
        QCheckBox {
            color: #d4d4d4;
        }
        """
        self.setStyleSheet(dark_style)
    
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        analyze_action = QAction("Analyze Markets", self)
        analyze_action.setShortcut("F5")
        analyze_action.triggered.connect(self.run_analysis)
        file_menu.addAction(analyze_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        dry_run_action = QAction("Dry Run Mode", self)
        dry_run_action.setCheckable(True)
        dry_run_action.setChecked(TRADING_MODE["dry_run"])
        settings_menu.addAction(dry_run_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the toolbar."""
        toolbar = QFrame()
        toolbar.setStyleSheet("QFrame { background-color: #2d2d30; border-bottom: 1px solid #3c3c3c; }")
        toolbar.setFixedHeight(60)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(16, 8, 16, 8)
        
        # Logo/Title
        title_label = QLabel("🤖 CHIBOY BOT")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4fc3f7;")
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        # Analyze button
        self.analyze_btn = QPushButton("🔄 Analyze Markets")
        self.analyze_btn.setFixedSize(160, 40)
        self.analyze_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return toolbar
    
    def create_opportunities_panel(self):
        """Create the opportunities list panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Header
        header = QLabel("🎯 Trading Opportunities")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")
        layout.addWidget(header)
        
        # Table
        self.opportunities_table = QTableWidget()
        self.opportunities_table.setColumnCount(7)
        self.opportunities_table.setHorizontalHeaderLabels([
            "Symbol", "Type", "Direction", "Timeframe", "Entry", "SL", "Confidence"
        ])
        self.opportunities_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.opportunities_table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.opportunities_table)
        
        return panel
    
    def create_details_panel(self):
        """Create the details and controls panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs
        tabs = QTabWidget()
        
        # Trade Details Tab
        trade_tab = QWidget()
        trade_layout = QVBoxLayout(trade_tab)
        
        # Selected Trade Info
        info_group = QGroupBox("Selected Opportunity")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Symbol:"), 0, 0)
        self.lbl_symbol = QLabel("--")
        self.lbl_symbol.setStyleSheet("font-size: 18px; font-weight: bold; color: #4fc3f7;")
        info_layout.addWidget(self.lbl_symbol, 0, 1)
        
        info_layout.addWidget(QLabel("Type:"), 1, 0)
        self.lbl_type = QLabel("--")
        info_layout.addWidget(self.lbl_type, 1, 1)
        
        info_layout.addWidget(QLabel("Direction:"), 2, 0)
        self.lbl_direction = QLabel("--")
        info_layout.addWidget(self.lbl_direction, 2, 1)
        
        info_layout.addWidget(QLabel("Confidence:"), 3, 0)
        self.lbl_confidence = QLabel("--")
        info_layout.addWidget(self.lbl_confidence, 3, 1)
        
        info_layout.addWidget(QLabel("Risk:Reward:"), 4, 0)
        self.lbl_rr = QLabel("--")
        info_layout.addWidget(self.lbl_rr, 4, 1)
        
        trade_layout.addWidget(info_group)
        
        # Price Levels
        levels_group = QGroupBox("Price Levels")
        levels_layout = QGridLayout(levels_group)
        
        levels_layout.addWidget(QLabel("Entry Price:"), 0, 0)
        self.lbl_entry = QLabel("--")
        levels_layout.addWidget(self.lbl_entry, 0, 1)
        
        levels_layout.addWidget(QLabel("Stop Loss:"), 1, 0)
        self.lbl_sl = QLabel("--")
        self.lbl_sl.setStyleSheet("color: #ef5350;")
        levels_layout.addWidget(self.lbl_sl, 1, 1)
        
        levels_layout.addWidget(QLabel("Take Profit:"), 2, 0)
        self.lbl_tp = QLabel("--")
        self.lbl_tp.setStyleSheet("color: #66bb6a;")
        levels_layout.addWidget(self.lbl_tp, 2, 1)
        
        trade_layout.addWidget(levels_group)
        
        # Trade Buttons
        btn_layout = QHBoxLayout()
        
        self.buy_btn = QPushButton("📈 BUY / LONG")
        self.buy_btn.setStyleSheet("background-color: #2e7d32;")
        self.buy_btn.setFixedHeight(50)
        self.buy_btn.setEnabled(False)
        self.buy_btn.clicked.connect(lambda: self.execute_trade("long"))
        btn_layout.addWidget(self.buy_btn)
        
        self.sell_btn = QPushButton("📉 SELL / SHORT")
        self.sell_btn.setStyleSheet("background-color: #c62828;")
        self.sell_btn.setFixedHeight(50)
        self.sell_btn.setEnabled(False)
        self.sell_btn.clicked.connect(lambda: self.execute_trade("short"))
        btn_layout.addWidget(self.sell_btn)
        
        trade_layout.addLayout(btn_layout)
        trade_layout.addStretch()
        
        tabs.addTab(trade_tab, "Trade")
        
        # Settings Tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Mode settings
        mode_group = QGroupBox("Trading Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.dry_run_check = QCheckBox("Dry Run Mode (Simulation)")
        self.dry_run_check.setChecked(TRADING_MODE["dry_run"])
        mode_layout.addWidget(self.dry_run_check)
        
        self.live_trade_check = QCheckBox("Enable Live Trading")
        self.live_trade_check.setChecked(TRADING_MODE["execute_trades"])
        mode_layout.addWidget(self.live_trade_check)
        
        settings_layout.addWidget(mode_group)
        
        # Risk settings
        risk_group = QGroupBox("Risk Management")
        risk_layout = QGridLayout(risk_group)
        
        risk_layout.addWidget(QLabel("Max Risk %:"), 0, 0)
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 10.0)
        self.risk_spin.setValue(2.0)
        self.risk_spin.setSuffix(" %")
        risk_layout.addWidget(self.risk_spin, 0, 1)
        
        risk_layout.addWidget(QLabel("Max Open Trades:"), 1, 0)
        self.max_trades_spin = QSpinBox()
        self.max_trades_spin.setRange(1, 10)
        self.max_trades_spin.setValue(3)
        risk_layout.addWidget(self.max_trades_spin, 1, 1)
        
        risk_layout.addWidget(QLabel("Min R:R Ratio:"), 2, 0)
        self.rr_spin = QDoubleSpinBox()
        self.rr_spin.setRange(1.0, 5.0)
        self.rr_spin.setValue(2.0)
        risk_layout.addWidget(self.rr_spin, 2, 1)
        
        settings_layout.addWidget(risk_group)
        
        # API Status
        api_group = QGroupBox("API Status")
        api_layout = QVBoxLayout(api_group)
        
        self.oanda_status = QLabel("❌ OANDA: Not Connected")
        api_layout.addWidget(self.oanda_status)
        
        self.binance_status = QLabel("❌ Binance: Not Connected")
        api_layout.addWidget(self.binance_status)
        
        settings_layout.addWidget(api_group)
        
        settings_layout.addStretch()
        
        tabs.addTab(settings_tab, "⚙️ Settings")
        
        # Console Tab
        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("font-family: monospace; font-size: 11px;")
        console_layout.addWidget(self.console)
        
        tabs.addTab(console_tab, "📝 Console")
        
        layout.addWidget(tabs)
        
        return panel
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Configure API keys in .env to enable live data")
    
    def log(self, message, level="info"):
        """Add message to console."""
        color = "#d4d4d4"
        if level == "error":
            color = "#ef5350"
        elif level == "success":
            color = "#66bb6a"
        elif level == "warning":
            color = "#ffa726"
        
        self.console.append(f'<span style="color: {color};">{message}</span>')
    
    def run_analysis(self):
        """Run market analysis."""
        if self.analysis_thread and self.analysis_thread.isRunning():
            return
        
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        self.log("🔄 Starting market analysis...")
        self.status_bar.showMessage("Analyzing markets...")
        
        self.analysis_thread = AnalysisThread()
        self.analysis_thread.progress.connect(lambda m: self.log(f"  {m}"))
        self.analysis_thread.finished.connect(self.on_analysis_complete)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.start()
    
    def on_analysis_complete(self, opportunities):
        """Handle analysis completion."""
        self.opportunities = opportunities
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        # Update table
        self.opportunities_table.setRowCount(len(opportunities))
        
        for i, opp in enumerate(opportunities):
            o = opp["opportunity"]
            
            self.opportunities_table.setItem(i, 0, QTableWidgetItem(opp["symbol"]))
            self.opportunities_table.setItem(i, 1, QTableWidgetItem(opp["type"].upper()))
            
            dir_item = QTableWidgetItem(o["direction"].upper())
            if o["direction"] == "long":
                dir_item.setForeground(QColor("#66bb6a"))
            else:
                dir_item.setForeground(QColor("#ef5350"))
            self.opportunities_table.setItem(i, 2, dir_item)
            
            self.opportunities_table.setItem(i, 3, QTableWidgetItem(opp["timeframe"]))
            
            entry = str(o.get("entry_price", "Market"))
            self.opportunities_table.setItem(i, 4, QTableWidgetItem(entry))
            
            sl = str(o.get("stop_loss", "N/A"))
            self.opportunities_table.setItem(i, 5, QTableWidgetItem(sl))
            
            conf_item = QTableWidgetItem(f'{o["confidence"]:.0f}%')
            conf_item.setForeground(QColor("#4fc3f7"))
            self.opportunities_table.setItem(i, 6, conf_item)
        
        self.opportunities_table.resizeColumnsToContents()
        
        count = len(opportunities)
        self.log(f"✅ Analysis complete: {count} opportunities found", "success")
        self.status_bar.showMessage(f"Found {count} trading opportunities")
    
    def on_analysis_error(self, error):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.log(f"❌ Error: {error}", "error")
        self.status_bar.showMessage("Analysis failed - check API keys")
    
    def on_selection_changed(self):
        """Handle table selection."""
        selected = self.opportunities_table.selectedIndexes()
        if not selected:
            return
        
        row = selected[0].row()
        if row < len(self.opportunities):
            opp = self.opportunities[row]
            o = opp["opportunity"]
            
            # Update labels
            self.lbl_symbol.setText(opp["symbol"])
            self.lbl_type.setText(opp["type"].upper())
            
            dir_text = o["direction"].upper()
            self.lbl_direction.setText(dir_text)
            self.lbl_direction.setStyleSheet(
                f"font-weight: bold; color: {'#66bb6a' if o['direction'] == 'long' else '#ef5350'};"
            )
            
            self.lbl_confidence.setText(f'{o["confidence"]:.0f}%')
            self.lbl_rr.setText(f'1:{o.get("risk_reward", 0):.1f}')
            
            self.lbl_entry.setText(str(o.get("entry_price", "Market")))
            self.lbl_sl.setText(str(o.get("stop_loss", "N/A")))
            self.lbl_tp.setText(str(o.get("take_profit", "N/A")))
            
            # Enable buttons
            self.buy_btn.setEnabled(True)
            self.sell_btn.setEnabled(True)
    
    def execute_trade(self, direction):
        """Execute a trade."""
        if not self.opportunities_table.selectedIndexes():
            return
        
        row = self.opportunities_table.selectedIndexes()[0].row()
        opp = self.opportunities[row]
        o = opp["opportunity"]
        
        mode = "DRY RUN" if self.dry_run_check.isChecked() else "LIVE"
        
        reply = QMessageBox.question(
            self, "Confirm Trade",
            f"Execute {direction.upper()} on {opp['symbol']}?\n\nMode: {mode}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.dry_run_check.isChecked():
                self.log(f"🎯 [DRY RUN] {direction.upper()} {opp['symbol']} @ {o.get('entry_price', 'market')}", "success")
                QMessageBox.information(self, "Trade Simulated", "Trade executed in DRY RUN mode")
            else:
                self.log(f"⚠️ [LIVE] {direction.upper()} {opp['symbol']} - Execution requires API keys", "warning")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About CHIBOY BOT",
            f"<h3>CHIBOY BOT v{BOT_CONFIG['version']}</h3>"
            "<p>A comprehensive trading bot implementing CHIBOY concepts.</p>"
            "<p>Supports OANDA (Forex) and Binance (Crypto)</p>"
            "<hr><p>© 2026</p>"
        )


# ==================== MAIN ====================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CHIBOY BOT")
    app.setApplicationVersion(BOT_CONFIG["version"])
    
    window = CHIBOYTradingBotGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
