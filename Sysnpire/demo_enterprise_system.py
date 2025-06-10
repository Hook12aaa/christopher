#!/usr/bin/env python3
"""
Demo Enterprise System - Simple demonstration of working enterprise architecture

Shows the complete enterprise field theory system in action.
"""

from model import ChargeFactory

def main():
    print("🏆 Enterprise Field Theory System Demo")
    print("=" * 50)
    
    # Initialize enterprise system
    print("🏭 Initializing Enterprise System...")
    factory = ChargeFactory()
    
    # Create enterprise charges
    print(f"\n📝 Creating Enterprise Charges...")
    
    enterprise_texts = [
        "Enterprise-grade semantic processing for commercial applications",
        "Field-theoretic universe management with persistent storage",
        "Commercial REST API for scalable text analysis services"
    ]
    
    # Process through enterprise pipeline
    charges = factory.create_charges_batch(enterprise_texts)
    
    # Show enterprise results
    print(f"\n📊 Enterprise Results:")
    for i, charge in enumerate(charges):
        print(f"   [{i+1}] {charge.charge_id}")
        print(f"       Text: '{charge.text_source[:60]}...'")
        print(f"       Charge: {charge.complete_charge}")
        print(f"       Position: {charge.metadata.field_position}")
        print(f"       Region: {charge.metadata.field_region}")
        print(f"       Nearby: {len(charge.metadata.nearby_charges)} charges")
    
    # Universe analytics
    metrics = factory.get_universe_metrics()
    
    print(f"\n🌌 Enterprise Universe Status:")
    print(f"   Total charges: {metrics['universe_metrics']['total_charges']}")
    print(f"   Total energy: {metrics['universe_metrics']['total_energy']:.6f}")
    print(f"   Field regions: {metrics['field_regions']}")
    print(f"   Database: {metrics['storage_path']}")
    
    # Query demonstration
    print(f"\n🔍 Enterprise Query Demo:")
    
    # Query by field region
    region = charges[0].metadata.field_region
    region_charges = factory.query_charges(field_region=region)
    print(f"   Charges in {region}: {len(region_charges)}")
    
    # Nearby charges
    nearby = factory.query_charges(nearby_charge_id=charges[0].charge_id)
    print(f"   Charges near {charges[0].charge_id}: {len(nearby)}")
    
    # Collective response
    charge_ids = [c.charge_id for c in charges]
    collective = factory.compute_collective_response(charge_ids)
    print(f"   Collective response: {abs(collective):.6f}")
    
    print(f"\n✅ Enterprise System Fully Operational!")
    print(f"🚀 Ready for commercial deployment")

if __name__ == "__main__":
    main()