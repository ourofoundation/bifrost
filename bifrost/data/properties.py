"""
Property discretization utilities for BIFROST.

This module handles converting continuous material properties into discrete bins
for use in the model's property prefix conditioning system.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class PropertyBinner:
    """Handles discretization of continuous properties into discrete bins."""

    def __init__(self):
        """Initialize property binning configuration."""
        self.property_bins = {
            "ehull": {
                "thresholds": [0.01, 0.05, 0.1],  # eV/atom
                "tokens": ["EHULL_NONE", "EHULL_LOW", "EHULL_MED", "EHULL_HIGH"],
                "description": "Energy above convex hull",
            },
            "bandgap": {
                "thresholds": [0.5, 2.0, 4.0],  # eV
                "tokens": [
                    "BANDGAP_NONE",
                    "BANDGAP_LOW",
                    "BANDGAP_MED",
                    "BANDGAP_HIGH",
                ],
                "description": "Electronic band gap",
            },
            "density": {
                "thresholds": [2.0, 4.0, 8.0],  # g/cm³
                "tokens": [
                    "DENSITY_NONE",
                    "DENSITY_LOW",
                    "DENSITY_MED",
                    "DENSITY_HIGH",
                ],
                "description": "Mass density",
            },
            "bulk_modulus": {
                "thresholds": [50, 150, 300],  # GPa
                "tokens": ["BULK_NONE", "BULK_LOW", "BULK_MED", "BULK_HIGH"],
                "description": "Bulk modulus",
            },
            "formation_energy": {
                "thresholds": [-2.0, -0.5, 0.0],  # eV/atom
                "tokens": ["FORM_NONE", "FORM_LOW", "FORM_MED", "FORM_HIGH"],
                "description": "Formation energy",
            },
        }

    def discretize_property(self, value: float, thresholds: List[float]) -> str:
        """
        Convert a continuous property value into a discrete bin.

        Args:
            value: Continuous property value (can be None for masked properties)
            thresholds: List of threshold values defining bin boundaries

        Returns:
            String representing the discrete bin ('NONE', 'LOW', 'MED', 'HIGH')
        """
        if value is None:
            return "NONE"
        if value < thresholds[0]:
            return "NONE"
        elif value < thresholds[1]:
            return "LOW"
        elif value < thresholds[2]:
            return "MED"
        else:
            return "HIGH"

    def get_property_bin(self, prop_name: str, value: Optional[float]) -> str:
        """
        Get the discrete bin for a specific property.

        Args:
            prop_name: Name of the property (e.g., 'bandgap', 'density')
            value: Continuous property value

        Returns:
            Discrete bin token string

        Raises:
            ValueError: If property name is not recognized
        """
        if prop_name not in self.property_bins:
            raise ValueError(f"Unknown property: {prop_name}")

        thresholds = self.property_bins[prop_name]["thresholds"]
        return self.discretize_property(value, thresholds)

    def get_property_token(self, prop_name: str, bin_name: str) -> str:
        """
        Get the full token for a property bin combination.

        Args:
            prop_name: Name of the property
            bin_name: Discrete bin name ('NONE', 'LOW', 'MED', 'HIGH')

        Returns:
            Full token string (e.g., 'BANDGAP_MED')
        """
        if prop_name not in self.property_bins:
            raise ValueError(f"Unknown property: {prop_name}")

        tokens = self.property_bins[prop_name]["tokens"]
        bin_index = ["NONE", "LOW", "MED", "HIGH"].index(bin_name)
        return tokens[bin_index]

    def get_all_property_tokens(self) -> List[str]:
        """Get list of all possible property tokens."""
        all_tokens = []
        for prop_config in self.property_bins.values():
            all_tokens.extend(prop_config["tokens"])
        return all_tokens

    def get_property_info(self, prop_name: str) -> Dict[str, Any]:
        """Get information about a property including thresholds and description."""
        if prop_name not in self.property_bins:
            raise ValueError(f"Unknown property: {prop_name}")
        return self.property_bins[prop_name].copy()

    def is_valid_property(self, prop_name: str) -> bool:
        """Check if a property name is valid."""
        return prop_name in self.property_bins


# Global instance for easy access
property_binner = PropertyBinner()


def discretize_structure_properties(properties: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert all properties in a structure to discrete bins.

    Args:
        properties: Dictionary of property_name -> continuous_value (values can be None for masked properties)

    Returns:
        Dictionary of property_name -> bin_token
    """
    discretized = {}
    for prop_name, value in properties.items():
        if property_binner.is_valid_property(prop_name) and value is not None:
            bin_name = property_binner.get_property_bin(prop_name, value)
            token = property_binner.get_property_token(prop_name, bin_name)
            discretized[prop_name] = token
    return discretized


def get_property_thresholds(prop_name: str) -> List[float]:
    """Get thresholds for a property."""
    return property_binner.property_bins[prop_name]["thresholds"]


def get_property_bins() -> Dict[str, Dict[str, Any]]:
    """Get the complete property binning configuration."""
    return property_binner.property_bins.copy()
