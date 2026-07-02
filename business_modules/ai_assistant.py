"""
AI-POWERED CHAT ASSISTANT FOR RETAIL ANALYTICS
----------------------------------------------
Answers questions about:
- Sales forecasts
- Inventory status
- Pricing recommendations
- GST calculations
"""

import pandas as pd
import numpy as np


class RetailAIAssistant:
    """
    AI-powered chat assistant for retail analytics
    Answers questions about sales, inventory, pricing, GST
    """
    
    def __init__(self, results_data):
        """
        Initialize with analysis results
        
        Parameters:
        results_data: dict - complete analysis results
        """
        self.data = results_data
        self.conversation_history = []
    
    def answer_question(self, question):
        """Main method to answer user questions"""
        
        question_lower = question.lower()
        
        # Question routing
        if any(word in question_lower for word in ['best', 'top', 'highest', 'most']):
            return self._handle_top_questions(question_lower)
        
        elif any(word in question_lower for word in ['worst', 'lowest', 'bottom', 'least']):
            return self._handle_bottom_questions(question_lower)
        
        elif any(word in question_lower for word in ['inventory', 'stock', 'reorder']):
            return self._handle_inventory_questions(question_lower)
        
        elif any(word in question_lower for word in ['price', 'pricing', 'discount']):
            return self._handle_pricing_questions(question_lower)
        
        elif any(word in question_lower for word in ['gst', 'tax']):
            return self._handle_gst_questions(question_lower)
        
        elif any(word in question_lower for word in ['forecast', 'predict', 'future', 'next']):
            return self._handle_forecast_questions(question_lower)
        
        else:
            return self._handle_general_questions(question_lower)
    
    def _handle_top_questions(self, question):
        """Handles questions about top/best performing items"""
        
        product_preds = self.data.get('product_predictions', {})
        
        if not product_preds:
            return "ℹ️ Product-level data is not available in your dataset."
        
        sorted_products = sorted(
            product_preds.items(),
            key=lambda x: x[1]['latest_prediction'],
            reverse=True
        )
        
        top_5 = sorted_products[:5]
        
        response = "🏆 **Top 5 Performing Products:**\n\n"
        for i, (product, data) in enumerate(top_5, 1):
            response += f"{i}. **{product}**: ₹{data['latest_prediction']:,.0f} predicted sales\n"
        
        return response
    
    def _handle_bottom_questions(self, question):
        """Handles questions about worst/lowest performing items"""
        
        product_preds = self.data.get('product_predictions', {})
        
        if not product_preds:
            return "ℹ️ Product-level data is not available."
        
        sorted_products = sorted(
            product_preds.items(),
            key=lambda x: x[1]['latest_prediction']
        )
        
        bottom_5 = sorted_products[:5]
        
        response = "⚠️ **Bottom 5 Performing Products:**\n\n"
        for i, (product, data) in enumerate(bottom_5, 1):
            response += f"{i}. **{product}**: ₹{data['latest_prediction']:,.0f} predicted sales\n"
        
        response += "\n💡 **Recommendation**: Consider promotional pricing or discontinuing these products."
        
        return response
    
    def _handle_inventory_questions(self, question):
        """Handles inventory-related questions"""
        
        inventory_report = self.data.get('inventory_report', {})
        
        if not inventory_report:
            return "ℹ️ Inventory data is not available."
        
        critical_items = {
            product: data for product, data in inventory_report.items()
            if data.get('status') == 'CRITICAL'
        }
        
        low_items = {
            product: data for product, data in inventory_report.items()
            if data.get('status') == 'LOW'
        }
        
        if 'critical' in question or 'urgent' in question:
            if not critical_items:
                return "✅ No critical stock items. All products have adequate inventory."
            
            response = "🚨 **Critical Stock Items:**\n\n"
            for product, data in critical_items.items():
                response += f"• **{product}**: {data['stock_coverage_days']:.1f} days left\n"
                response += f"  Action: {data['action']}\n\n"
            
            return response
        
        elif 'low' in question or 'reorder' in question:
            all_low = {**critical_items, **low_items}
            
            if not all_low:
                return "✅ All products have sufficient stock levels."
            
            response = "📦 **Products Needing Reorder:**\n\n"
            for product, data in all_low.items():
                response += f"• **{product}**: {data['current_stock']:.0f} units ({data['stock_coverage_days']:.1f} days)\n"
            
            return response
        
        else:
            total_items = len(inventory_report)
            critical_count = len(critical_items)
            low_count = len(low_items)
            
            response = f"📊 **Inventory Summary:**\n\n"
            response += f"• Total Products: {total_items}\n"
            response += f"• Critical Stock: {critical_count}\n"
            response += f"• Low Stock: {low_count}\n"
            response += f"• Healthy Stock: {total_items - critical_count - low_count}\n"
            
            return response
    
    def _handle_pricing_questions(self, question):
        """Handles pricing-related questions"""
        
        pricing_report = self.data.get('pricing_report', {})
        
        if not pricing_report:
            return "ℹ️ Pricing data is not available."
        
        reduce_price = [
            (product, data) for product, data in pricing_report.items()
            if data.get('suggested_action') == 'REDUCE PRICE'
        ]
        
        increase_price = [
            (product, data) for product, data in pricing_report.items()
            if data.get('suggested_action') == 'INCREASE PRICE'
        ]
        
        if 'reduce' in question or 'discount' in question:
            if not reduce_price:
                return "✅ No products currently need price reduction."
            
            response = "💰 **Products to Consider Discounting:**\n\n"
            for product, data in reduce_price[:5]:
                response += f"• **{product}**\n"
                response += f"  Current: ₹{data['current_price']:.2f} → Suggested: ₹{data['adjusted_price']:.2f}\n"
                response += f"  Reason: {data['reason']}\n\n"
            
            return response
        
        elif 'increase' in question:
            if not increase_price:
                return "⚠️ No products currently suitable for price increase."
            
            response = "📈 **Products Ready for Price Increase:**\n\n"
            for product, data in increase_price[:5]:
                response += f"• **{product}**\n"
                response += f"  Current: ₹{data['current_price']:.2f}\n"
                response += f"  Reason: {data['reason']}\n\n"
            
            return response
        
        else:
            response = "💰 **Pricing Strategy Summary:**\n\n"
            response += f"• Reduce Price: {len(reduce_price)} products\n"
            response += f"• Increase Price: {len(increase_price)} products\n"
            response += f"• Maintain Price: {len(pricing_report) - len(reduce_price) - len(increase_price)} products\n"
            
            return response
    
    def _handle_gst_questions(self, question):
        """Handles GST-related questions"""
        
        gst_report = self.data.get('gst_report')
        
        if not gst_report:
            return "ℹ️ GST data is not available."
        
        if isinstance(gst_report, dict) and len(gst_report) > 0:
            total_gst = sum(data.get('net_gst_payable', 0) for data in gst_report.values())
            
            response = "🧾 **GST Summary:**\n\n"
            response += f"• Total Net GST Payable: ₹{total_gst:,.2f}\n"
            response += f"• Number of Products: {len(gst_report)}\n"
            
            top_gst = sorted(
                gst_report.items(),
                key=lambda x: x[1].get('net_gst_payable', 0),
                reverse=True
            )[:3]
            
            response += "\n**Top GST Contributors:**\n"
            for product, data in top_gst:
                response += f"• {product}: ₹{data.get('net_gst_payable', 0):,.2f}\n"
            
            return response
        
        return "ℹ️ GST calculation data not available."
    
    def _handle_forecast_questions(self, question):
        """Handles forecasting questions"""
        
        prophet_forecast = self.data.get('prophet_forecast')
        timeline = self.data.get('timeline', 'period')
        
        if prophet_forecast is None or len(prophet_forecast) == 0:
            return "ℹ️ Forecast data is not available."
        
        next_period = prophet_forecast.iloc[0]
        
        response = "🔮 **Sales Forecast:**\n\n"
        response += f"• Next {timeline}: ₹{next_period['Prophet_Forecast']:,.0f}\n"
        
        total_forecast = prophet_forecast['Prophet_Forecast'].sum()
        response += f"• Total {len(prophet_forecast)}-{timeline} Forecast: ₹{total_forecast:,.0f}\n"
        
        avg_forecast = prophet_forecast['Prophet_Forecast'].mean()
        response += f"• Average per {timeline}: ₹{avg_forecast:,.0f}\n"
        
        return response
    
    def _handle_general_questions(self, question):
        """Handles general questions"""
        
        suggestions = [
            "📊 Try asking:",
            "• What are my top performing products?",
            "• Which products need reordering?",
            "• What's my sales forecast?",
            "• Which products should I discount?",
            "• What's my total GST liability?",
            "• Show me low stock items"
        ]
        
        return "\n".join(suggestions)
    
    def get_smart_insights(self):
        """
        Generates automatic AI insights without being asked
        """
        
        insights = []
        
        # Check critical inventory
        inventory_report = self.data.get('inventory_report', {})
        critical_items = [
            product for product, data in inventory_report.items()
            if data.get('status') == 'CRITICAL'
        ]
        
        if critical_items:
            insights.append({
                'type': 'alert',
                'title': '🚨 Urgent Action Required',
                'message': f"{len(critical_items)} products have critical stock levels: {', '.join(critical_items[:3])}"
            })
        
        # Check high-performing products
        product_preds = self.data.get('product_predictions', {})
        if product_preds:
            top_product = max(product_preds.items(), key=lambda x: x[1]['latest_prediction'])
            insights.append({
                'type': 'success',
                'title': '🏆 Star Product',
                'message': f"{top_product[0]} is your top performer with ₹{top_product[1]['latest_prediction']:,.0f} predicted sales"
            })
        
        # Check pricing opportunities
        pricing_report = self.data.get('pricing_report', {})
        if pricing_report:
            can_increase = [
                p for p, d in pricing_report.items()
                if d.get('suggested_action') == 'INCREASE PRICE'
            ]
            
            if can_increase:
                insights.append({
                    'type': 'opportunity',
                    'title': '💰 Revenue Opportunity',
                    'message': f"{len(can_increase)} products can support price increases without hurting demand"
                })
        
        return insights