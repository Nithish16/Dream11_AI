#!/usr/bin/env python3
"""
Performance Optimizer - Tournament Winning Analytics
Advanced team analysis for maximum Dream11 success
"""

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import statistics

@dataclass
class TeamPerformanceMetrics:
    """Advanced team performance analytics"""
    value_score: float
    differential_potential: float
    captain_upside: float
    consistency_rating: float
    tournament_suitability: str
    contest_recommendation: str
    risk_reward_ratio: float
    ownership_leverage: float
    key_advantages: List[str]
    potential_concerns: List[str]

class PerformanceOptimizer:
    """Optimize team performance for maximum tournament success"""
    
    def __init__(self):
        self.success_multipliers = {
            'captain_performance': 2.0,
            'vice_captain_performance': 1.5,
            'differential_value': 1.3,
            'consistency_bonus': 1.2
        }
    
    def analyze_team_performance(self, team, contest_type: str = "gpp") -> TeamPerformanceMetrics:
        """
        Comprehensive team performance analysis
        
        Args:
            team: OptimalTeam object
            contest_type: 'gpp' for tournaments, 'cash' for cash games
            
        Returns:
            TeamPerformanceMetrics with actionable insights
        """
        
        # Calculate value score (points per credit efficiency)
        total_expected_points = sum(p.final_score for p in team.players)
        if team.captain:
            total_expected_points += team.captain.final_score  # Captain bonus
        if team.vice_captain:
            total_expected_points += team.vice_captain.final_score * 0.5  # VC bonus
            
        value_score = total_expected_points / max(team.total_credits, 1.0)
        
        # Calculate differential potential
        ownership_scores = [getattr(p, 'ownership_prediction', 50.0) for p in team.players]
        avg_ownership = statistics.mean(ownership_scores)
        differential_potential = (100 - avg_ownership) / 100 * total_expected_points
        
        # Captain upside analysis
        captain_upside = 0
        if team.captain:
            captain_score = team.captain.final_score
            captain_ownership = getattr(team.captain, 'ownership_prediction', 50.0)
            captain_upside = captain_score * 2 * (100 - captain_ownership) / 100
        
        # Consistency rating
        consistency_scores = [p.consistency_score for p in team.players]
        consistency_rating = statistics.mean(consistency_scores)
        
        # Tournament suitability
        tournament_suitability = self._assess_tournament_suitability(
            avg_ownership, consistency_rating, differential_potential
        )
        
        # Contest recommendation
        contest_recommendation = self._get_contest_recommendation(
            avg_ownership, consistency_rating, value_score, contest_type
        )
        
        # Risk-reward ratio
        risk_factors = [100 - p.consistency_score for p in team.players]
        avg_risk = statistics.mean(risk_factors)
        risk_reward_ratio = total_expected_points / max(avg_risk, 1.0)
        
        # Ownership leverage
        differential_values = [getattr(p, 'differential_value', 0.0) for p in team.players]
        ownership_leverage = statistics.mean(differential_values)
        
        # Key advantages and concerns
        key_advantages = self._identify_key_advantages(team, contest_type)
        potential_concerns = self._identify_concerns(team)
        
        return TeamPerformanceMetrics(
            value_score=value_score,
            differential_potential=differential_potential,
            captain_upside=captain_upside,
            consistency_rating=consistency_rating,
            tournament_suitability=tournament_suitability,
            contest_recommendation=contest_recommendation,
            risk_reward_ratio=risk_reward_ratio,
            ownership_leverage=ownership_leverage,
            key_advantages=key_advantages,
            potential_concerns=potential_concerns
        )
    
    def _assess_tournament_suitability(self, avg_ownership: float, 
                                     consistency: float, 
                                     differential: float) -> str:
        """Assess how well team suits tournaments"""
        
        if avg_ownership < 25 and differential > 500:
            return "🚀 EXCELLENT - High upside, low ownership"
        elif avg_ownership < 35 and consistency > 70:
            return "✅ VERY GOOD - Balanced risk/reward"
        elif avg_ownership < 50:
            return "👍 GOOD - Decent tournament potential"
        elif consistency > 80:
            return "🛡️ SAFE - Better for cash games"
        else:
            return "⚠️ RISKY - High ownership, uncertain upside"
    
    def _get_contest_recommendation(self, avg_ownership: float, 
                                  consistency: float, 
                                  value_score: float,
                                  contest_type: str) -> str:
        """Get specific contest recommendations"""
        
        if contest_type == "gpp":
            if avg_ownership < 30 and value_score > 50:
                return "🎯 GPP OPTIMAL - Use in large tournaments"
            elif avg_ownership < 45:
                return "🎲 GPP VIABLE - Good for mid-stake tournaments"
            else:
                return "💰 CASH BETTER - Too chalky for tournaments"
        else:
            if consistency > 75 and avg_ownership > 50:
                return "💰 CASH OPTIMAL - Safe for 50/50s and H2H"
            elif consistency > 65:
                return "💰 CASH VIABLE - Good for double-ups"
            else:
                return "🎯 GPP BETTER - Too risky for cash games"
    
    def _identify_key_advantages(self, team, contest_type: str) -> List[str]:
        """Identify team's key competitive advantages"""
        advantages = []
        
        # Captain analysis
        if team.captain:
            captain_ownership = getattr(team.captain, 'ownership_prediction', 50.0)
            if captain_ownership < 20:
                advantages.append(f"🎯 Contrarian Captain: {team.captain.name} (<{captain_ownership:.0f}% owned)")
            elif captain_ownership > 70:
                advantages.append(f"👑 Premium Captain: {team.captain.name} (proven performer)")
        
        # Value picks
        value_picks = []
        for player in team.players:
            value_ratio = player.final_score / max(player.credits, 1.0)
            if value_ratio > 55:  # High value threshold
                value_picks.append(player.name)
        
        if value_picks:
            advantages.append(f"💎 Value Picks: {', '.join(value_picks[:2])}")
        
        # Role balance
        role_counts = {'bat': 0, 'bowl': 0, 'ar': 0, 'wk': 0}
        for player in team.players:
            role = player.role.lower()
            if 'bat' in role:
                role_counts['bat'] += 1
            elif 'bowl' in role:
                role_counts['bowl'] += 1
            elif 'allrounder' in role:
                role_counts['ar'] += 1
            elif 'wk' in role or 'keeper' in role:
                role_counts['wk'] += 1
        
        if role_counts['ar'] >= 3:
            advantages.append("⚖️ All-rounder Heavy: Multiple scoring avenues")
        
        # Differential potential
        low_owned_players = [p for p in team.players if getattr(p, 'ownership_prediction', 50.0) < 25]
        if len(low_owned_players) >= 4:
            advantages.append(f"📈 High Differential: {len(low_owned_players)} low-owned players")
        
        return advantages[:4]  # Limit to top 4 advantages
    
    def _identify_concerns(self, team) -> List[str]:
        """Identify potential team concerns"""
        concerns = []
        
        # High ownership concerns
        high_owned = [p for p in team.players if getattr(p, 'ownership_prediction', 50.0) > 70]
        if len(high_owned) >= 6:
            concerns.append(f"⚠️ Too Chalky: {len(high_owned)} highly owned players")
        
        # Low consistency concerns
        risky_players = [p for p in team.players if p.consistency_score < 40]
        if len(risky_players) >= 3:
            risky_names = [p.name for p in risky_players[:2]]
            concerns.append(f"🎲 High Variance: {', '.join(risky_names)} inconsistent")
        
        # Budget concerns
        if team.total_credits < 90:
            concerns.append("💰 Underpriced: May lack premium options")
        elif team.total_credits > 99:
            concerns.append("💸 Overpriced: Very tight budget constraints")
        
        # Captain concerns
        if team.captain:
            captain_ownership = getattr(team.captain, 'ownership_prediction', 50.0)
            if captain_ownership > 80:
                concerns.append(f"👑 Chalky Captain: {team.captain.name} very highly owned")
        
        return concerns[:3]  # Limit to top 3 concerns
    
    def generate_actionable_summary(self, team, metrics: TeamPerformanceMetrics) -> str:
        """Generate actionable team summary"""
        
        summary_parts = []
        
        # Overall rating
        if metrics.value_score > 55 and metrics.differential_potential > 400:
            rating = "🏆 TOURNAMENT WINNER"
        elif metrics.value_score > 50:
            rating = "✅ STRONG CONTENDER" 
        elif metrics.consistency_rating > 75:
            rating = "🛡️ SAFE CHOICE"
        else:
            rating = "⚠️ HIGH RISK"
        
        summary_parts.append(f"{rating}")
        summary_parts.append(f"Value: {metrics.value_score:.1f} pts/credit")
        summary_parts.append(f"Leverage: {metrics.ownership_leverage:.0f}%")
        summary_parts.append(metrics.contest_recommendation)
        
        return " | ".join(summary_parts)

def format_optimized_team_output(team, team_num: int, contest_type: str = "gpp") -> str:
    """
    Generate optimized team output focused on winning
    """
    optimizer = PerformanceOptimizer()
    metrics = optimizer.analyze_team_performance(team, contest_type)
    
    output = []
    output.append("=" * 80)
    output.append(f"🏆 TEAM {team_num} - {team.strategy.upper()}")
    output.append("=" * 80)
    
    # Performance summary
    summary = optimizer.generate_actionable_summary(team, metrics)
    output.append(f"📊 {summary}")
    output.append("")
    
    # Captain & VC with performance indicators
    captain_info = f"👑 Captain: {team.captain.name}" if team.captain else "👑 Captain: None"
    vc_info = f"🥈 Vice: {team.vice_captain.name}" if team.vice_captain else "🥈 Vice: None"
    
    if team.captain:
        captain_ownership = getattr(team.captain, 'ownership_prediction', 50.0)
        captain_indicator = "🎯" if captain_ownership < 30 else "👑" if captain_ownership > 60 else "⚖️"
        captain_info += f" {captain_indicator}({captain_ownership:.0f}%)"
    
    output.append(f"{captain_info} | {vc_info}")
    output.append(f"💰 Budget: {team.total_credits:.1f}/100 | 🧠 Expected: {metrics.value_score * team.total_credits:.0f} pts")
    output.append("")
    
    # Key advantages
    if metrics.key_advantages:
        output.append("✅ KEY ADVANTAGES:")
        for advantage in metrics.key_advantages:
            output.append(f"   {advantage}")
        output.append("")
    
    # Potential concerns
    if metrics.potential_concerns:
        output.append("⚠️ WATCH OUT FOR:")
        for concern in metrics.potential_concerns:
            output.append(f"   {concern}")
        output.append("")
    
    # Player list with value indicators
    output.append("👥 LINEUP:")
    role_sections = {
        'Batsmen': [p for p in team.players if 'bat' in p.role.lower() and 'wk' not in p.role.lower()],
        'Keepers': [p for p in team.players if 'wk' in p.role.lower() or 'keeper' in p.role.lower()],
        'All-Rounders': [p for p in team.players if 'allrounder' in p.role.lower()],
        'Bowlers': [p for p in team.players if 'bowl' in p.role.lower() and 'allrounder' not in p.role.lower()]
    }
    
    for role_name, players in role_sections.items():
        if players:
            output.append(f"   {role_name}:")
            for player in players:
                value_ratio = player.final_score / max(player.credits, 1.0)
                ownership = getattr(player, 'ownership_prediction', 50.0)
                
                # Value indicator
                if value_ratio > 55:
                    value_icon = "💎"  # High value
                elif value_ratio > 50:
                    value_icon = "✅"  # Good value  
                elif value_ratio < 45:
                    value_icon = "⚠️"  # Poor value
                else:
                    value_icon = "⚖️"  # Average value
                
                # Ownership indicator
                if ownership < 25:
                    own_icon = "🎯"  # Low owned
                elif ownership > 70:
                    own_icon = "🔥"  # High owned
                else:
                    own_icon = ""
                
                captain_mark = "👑" if team.captain and player.name == team.captain.name else ""
                vc_mark = "🥈" if team.vice_captain and player.name == team.vice_captain.name else ""
                
                output.append(f"     {value_icon} {player.name} ({player.credits}c) {own_icon}{captain_mark}{vc_mark}")
    
    output.append("")
    output.append(f"🎯 Strategy: {metrics.tournament_suitability}")
    output.append("=" * 80)
    
    return "\n".join(output)