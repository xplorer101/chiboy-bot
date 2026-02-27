"""
CHIBOY BOT - Android/Kivy Version
=================================
A mobile-friendly trading bot using Kivy framework.
"""

import os
import sys
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.checkbox import CheckBox
from kivy.core.window import Window
from kivy.properties import StringProperty, ListProperty
from kivy.clock import Clock
import threading

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import BOT_CONFIG, TRADING_MODE
from src.analysis.multi_timeframe import MultiTimeframeAnalyzer
from src.signals.trading_signals import TradingSignals


class AnalysisThread(threading.Thread):
    """Background thread for market analysis."""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.opportunities = []
        
    def run(self):
        try:
            analyzer = MultiTimeframeAnalyzer()
            self.opportunities = analyzer.get_top_opportunities(limit=10)
            Clock.schedule_once(lambda dt: self.callback(self.opportunities))
        except Exception as e:
            Clock.schedule_once(lambda dt: self.callback([]))


class MainScreen(Screen):
    """Main dashboard screen."""
    
    opportunities = ListProperty([])
    status_text = StringProperty("Ready - Tap 'Analyze' to scan markets")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
        
    def build_ui(self):
        """Build the main UI."""
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Header
        header = BoxLayout(size_hint_y=None, height=60)
        title = Label(text='[b]🤖 CHIBOY BOT[/b]', font_size=24, markup=True)
        header.add_widget(title)
        layout.add_widget(header)
        
        # Status
        self.status_label = Label(text=self.status_text, size_hint_y=None, height=40)
        layout.add_widget(self.status_label)
        
        # Analyze Button
        self.analyze_btn = Button(text='🔄 ANALYZE MARKETS', 
                                 size_hint_y=None, height=50,
                                 background_color=(0.05, 0.3, 0.6, 1))
        self.analyze_btn.bind(on_press=self.run_analysis)
        layout.add_widget(self.analyze_btn)
        
        # Opportunities List
        scroll = ScrollView()
        self.opportunities_layout = GridLayout(cols=1, spacing=10, padding=10, size_hint_y=None)
        self.opportunities_layout.bind(minimum_height=self.opportunities_layout.setter('height'))
        scroll.add_widget(self.opportunities_layout)
        layout.add_widget(scroll)
        
        # Action Buttons
        action_layout = BoxLayout(size_hint_y=None, height=60, spacing=10)
        
        self.buy_btn = Button(text='📈 BUY', background_color=(0.2, 0.5, 0.2, 1))
        self.buy_btn.bind(on_press=self.buy)
        action_layout.add_widget(self.buy_btn)
        
        self.sell_btn = Button(text='📉 SELL', background_color=(0.5, 0.2, 0.2, 1))
        self.sell_btn.bind(on_press=self.sell)
        action_layout.add_widget(self.sell_btn)
        
        layout.add_widget(action_layout)
        
        # Settings
        settings_btn = Button(text='⚙️ SETTINGS', size_hint_y=None, height=40,
                            background_color=(0.3, 0.3, 0.3, 1))
        settings_btn.bind(on_press=self.go_to_settings)
        layout.add_widget(settings_btn)
        
        self.add_widget(layout)
    
    def run_analysis(self, *args):
        """Run market analysis."""
        self.status_text = "Analyzing markets..."
        self.analyze_btn.disabled = True
        self.analyze_btn.text = "⏳ SCANNING..."
        
        thread = AnalysisThread(self.on_analysis_complete)
        thread.start()
    
    def on_analysis_complete(self, opportunities):
        """Handle analysis completion."""
        self.opportunities = opportunities
        self.analyze_btn.disabled = False
        self.analyze_btn.text = "🔄 ANALYZE MARKETS"
        
        if opportunities:
            self.status_text = f"Found {len(opportunities)} opportunities!"
            self.update_opportunities_list()
        else:
            self.status_text = "No opportunities found"
    
    def update_opportunities_list(self):
        """Update the opportunities list display."""
        self.opportunities_layout.clear_widgets()
        
        for opp in self.opportunities:
            o = opp['opportunity']
            
            card = BoxLayout(orientation='vertical', size_hint_y=None, height=120,
                           padding=10, spacing=5)
            card.add_widget(Label(text=f"[b]{opp['symbol']}[/b] {opp['type'].upper()}", 
                               markup=True, size_hint_y=None, height=30))
            
            direction = o['direction'].upper()
            color = '[color=00ff00]' if direction == 'LONG' else '[color=ff0000]'
            card.add_widget(Label(text=f"{color}{direction}[/color] | {opp['timeframe']} | {o['confidence']:.0f}%",
                                markup=True, size_hint_y=None, height=25))
            
            card.add_widget(Label(text=f"Entry: {o.get('entry_price', 'Market')} | SL: {o.get('stop_loss', 'N/A')}",
                                size_hint_y=None, height=25))
            
            self.opportunities_layout.add_widget(card)
    
    def buy(self, *args):
        self.status_text = "BUY order placed (simulation)"
    
    def sell(self, *args):
        self.status_text = "SELL order placed (simulation)"
    
    def go_to_settings(self, *args):
        self.manager.current = 'settings'


class SettingsScreen(Screen):
    """Settings screen."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)
        
        # Title
        layout.add_widget(Label(text='[b]⚙️ Settings[/b]', font_size=24, markup=True, size_hint_y=None, height=50))
        
        # Dry Run
        dry_run_layout = BoxLayout(size_hint_y=None, height=40)
        dry_run_layout.add_widget(Label(text='Dry Run Mode'))
        self.dry_run_check = CheckBox(active=TRADING_MODE['dry_run'])
        dry_run_layout.add_widget(self.dry_run_check)
        layout.add_widget(dry_run_layout)
        
        # API Keys Section
        layout.add_widget(Label(text='[b]API Configuration[/b]', markup=True, size_hint_y=None, height=40))
        
        layout.add_widget(Label(text='OANDA API Key:', size_hint_y=None, height=30))
        self.oanda_key_input = TextInput(multiline=False, hint_text='Enter OANDA API key')
        layout.add_widget(self.oanda_key_input)
        
        layout.add_widget(Label(text='Binance API Key:', size_hint_y=None, height=30))
        self.binance_key_input = TextInput(multiline=False, hint_text='Enter Binance API key')
        layout.add_widget(self.binance_key_input)
        
        # Back Button
        layout.add_widget(Label())  # Spacer
        back_btn = Button(text='← BACK', size_hint_y=None, height=50,
                         background_color=(0.3, 0.3, 0.3, 1))
        back_btn.bind(on_press=self.go_back)
        layout.add_widget(back_btn)
        
        self.add_widget(layout)
    
    def go_back(self, *args):
        self.manager.current = 'main'


class ChiboyBotApp(App):
    """CHIBOY BOT Android Application."""
    
    def build(self):
        # Window configuration for Android
        Window.clearcolor = (0.1, 0.1, 0.15, 1)
        
        # Create screen manager
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(SettingsScreen(name='settings'))
        
        return sm
    
    def on_pause(self):
        return True


if __name__ == '__main__':
    ChiboyBotApp().run()
