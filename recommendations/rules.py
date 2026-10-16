"""
Domain Business Rule Catalog for AGENTIC SCOUT.
Maps analytical indicators and key driver patterns to actionable strategic guidance.
"""
from typing import Dict, Any, List


class BusinessRules:
    """Provides validated business playbooks for pricing, customer retention, and operations."""

    @staticmethod
    def get_controllable_levers() -> Dict[str, Dict[str, str]]:
        """Known business levers that decision makers can proactively adjust."""
        return {
            "marketing_spend": {
                "category": "Marketing & Growth",
                "lever": "Marketing Budget Allocation",
                "action_high": "Audit campaign channel ROI; reallocate underperforming spend to top-performing cohorts.",
                "action_low": "Test incremental marketing spend in high-converting segments to capture latent demand."
            },
            "discount_pct": {
                "category": "Pricing & Margin",
                "lever": "Discounting Strategy",
                "action_high": "Curtail blanket discounts; migrate toward targeted, performance-based rebates to protect margin.",
                "action_low": "Deploy selective promotional incentives to price-sensitive customer segments."
            },
            "support_tickets": {
                "category": "Customer Success & Operations",
                "lever": "Resolution & Support Capacity",
                "action_high": "Escalate support workflows; implement early-warning alerts for accounts logging >2 tickets in 30 days.",
                "action_low": "Maintain high first-contact resolution standards; monitor CSAT/NPS continuously."
            },
            "contract_months": {
                "category": "Contract & Retention Strategy",
                "lever": "Contract Commitment",
                "action_high": "Incentivize long-term contract conversions (annual/multi-year) through onboarding perks.",
                "action_low": "Provide friction-free renewal incentives for month-to-month cohorts approaching risk windows."
            },
            "monthly_charges": {
                "category": "Packaging & Value Perception",
                "lever": "Subscription Pricing & Tiering",
                "action_high": "Audit customer perceived value relative to pricing tier; offer customized feature bundles.",
                "action_low": "Explore value-add premium feature expansions for highly engaged accounts."
            },
            "promotion_active": {
                "category": "Promotional Planning",
                "lever": "Promotional Calendaring",
                "action_high": "Synchronize promotional flight dates with peak historical foot-traffic and seasonality.",
                "action_low": "Stagger promotional bursts to avoid demand cannibalization across adjacent periods."
            }
        }

    @staticmethod
    def match_lever(feature_name: str) -> Dict[str, str]:
        """Matches a feature to a business lever if available."""
        clean_name = str(feature_name).lower().replace(" ", "_").replace("-", "_")
        levers = BusinessRules.get_controllable_levers()
        
        for key, details in levers.items():
            if key in clean_name or clean_name in key:
                return details
                
        return {
            "category": "Operational Attribute",
            "lever": f"Manage {feature_name}",
            "action_high": f"Monitor and benchmark {feature_name} variations against target KPIs.",
            "action_low": f"Evaluate policy adjustments affecting {feature_name} across business units."
        }
