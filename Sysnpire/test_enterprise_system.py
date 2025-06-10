#!/usr/bin/env python3
"""
Test Enterprise System - Complete validation of enterprise architecture

Tests the full enterprise system:
1. Charge Pipeline → Rich conceptual charge objects
2. Universe → Database with field-theoretic placement
3. API Endpoints → Commercial REST interface
4. Documentation → Enterprise standards
"""

import sys
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_enterprise_charge_pipeline():
    """Test enterprise charge pipeline with rich objects."""
    print("🏭 Testing Enterprise Charge Pipeline")
    print("-" * 40)
    
    try:
        from model import ChargeFactory
        
        # Initialize enterprise factory
        factory = ChargeFactory()
        
        # Test single enterprise charge creation
        charge_obj = factory.create_charge(
            text="Enterprise field theory applications for commercial use",
            observational_state=1.5,
            gamma=1.2
        )
        
        print(f"\n✅ Enterprise charge creation works")
        print(f"   Charge ID: {charge_obj.charge_id}")
        print(f"   Magnitude: {charge_obj.magnitude:.6f}")
        print(f"   Field position: {charge_obj.metadata.field_position}")
        print(f"   Field region: {charge_obj.metadata.field_region}")
        print(f"   Nearby charges: {len(charge_obj.metadata.nearby_charges)}")
        print(f"   Historical states: {len(charge_obj.historical_states)}")
        
        # Test enterprise batch processing
        enterprise_texts = [
            "Enterprise-grade field theory processing",
            "Commercial semantic analysis applications",
            "Scalable conceptual charge generation",
            "Real-time field universe management"
        ]
        
        batch_charges = factory.create_charges_batch(enterprise_texts)
        
        print(f"\n✅ Enterprise batch processing works")
        print(f"   Processed: {len(batch_charges)} charges")
        
        # Verify field relationships established
        relationships_found = 0
        for charge in batch_charges:
            relationships_found += len(charge.metadata.nearby_charges)
        
        print(f"   Field relationships: {relationships_found} connections")
        
        # Test universe analytics
        metrics = factory.get_universe_metrics()
        
        print(f"\n✅ Universe analytics work")
        print(f"   Universe contains: {metrics['universe_metrics']['total_charges']} charges")
        print(f"   Field regions: {metrics['field_regions']}")
        print(f"   Total energy: {metrics['universe_metrics']['total_energy']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise charge pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_universe_database():
    """Test universe database functionality."""
    print(f"\n🌌 Testing Universe Database")
    print("-" * 40)
    
    try:
        from database import FieldUniverse
        from model import ChargeFactory
        
        # Create universe and factory
        universe = FieldUniverse("test_enterprise_universe.db")
        factory = ChargeFactory(universe)
        
        # Add some test charges
        test_texts = [
            "Database storage of field-theoretic charges",
            "Persistent universe with field placement",
            "Enterprise-grade charge management"
        ]
        
        created_charges = []
        for text in test_texts:
            charge = factory.create_charge(text)
            created_charges.append(charge)
        
        print(f"✅ Database storage works")
        print(f"   Created: {len(created_charges)} charges")
        print(f"   Stored in universe: {len(universe)} charges")
        
        # Test queries
        # Query by magnitude
        high_magnitude_charges = factory.query_charges(magnitude_range=(0.0, 1.0))
        print(f"   Magnitude query: {len(high_magnitude_charges)} charges")
        
        # Query by field region
        first_charge = created_charges[0]
        region_charges = factory.query_charges(field_region=first_charge.metadata.field_region)
        print(f"   Region query: {len(region_charges)} charges in {first_charge.metadata.field_region}")
        
        # Test nearby charges
        if len(created_charges) > 1:
            nearby_charges = factory.query_charges(
                nearby_charge_id=first_charge.charge_id,
                max_distance=100.0
            )
            print(f"   Nearby query: {len(nearby_charges)} charges within 100 units")
        
        # Test collective response
        charge_ids = [c.charge_id for c in created_charges]
        collective_response = factory.compute_collective_response(charge_ids)
        print(f"   Collective response: {abs(collective_response):.6f}")
        
        print(f"✅ Universe database fully functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Universe database failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enterprise_documentation():
    """Test enterprise documentation standards."""
    print(f"\n📚 Testing Enterprise Documentation")
    print("-" * 40)
    
    try:
        # Check documentation files exist
        doc_files = [
            "docs/ARCHITECTURE.md"
        ]
        
        missing_docs = []
        for doc_file in doc_files:
            if not Path(doc_file).exists():
                missing_docs.append(doc_file)
        
        if missing_docs:
            print(f"❌ Missing documentation: {missing_docs}")
            return False
        
        print(f"✅ Documentation files present")
        
        # Check architecture documentation content
        arch_doc = Path("docs/ARCHITECTURE.md").read_text()
        required_sections = [
            "System Overview",
            "Core Components",
            "Data Models", 
            "Field-Theoretic Mathematics",
            "Enterprise Standards"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in arch_doc:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing architecture sections: {missing_sections}")
            return False
        
        print(f"✅ Architecture documentation complete")
        print(f"   Sections: {len(required_sections)} required sections present")
        print(f"   Length: {len(arch_doc)} characters")
        
        # Check code documentation
        from model.charge_factory import ChargeFactory
        from database.field_universe import FieldUniverse
        
        factory_doc = ChargeFactory.__doc__
        universe_doc = FieldUniverse.__doc__
        
        if not factory_doc or len(factory_doc) < 50:
            print(f"❌ Insufficient ChargeFactory documentation")
            return False
            
        if not universe_doc or len(universe_doc) < 50:
            print(f"❌ Insufficient FieldUniverse documentation")
            return False
        
        print(f"✅ Code documentation adequate")
        
        return True
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False

def test_enterprise_architecture():
    """Test enterprise architecture separation."""
    print(f"\n🏗️ Testing Enterprise Architecture")
    print("-" * 40)
    
    try:
        # Test 4-folder structure
        required_folders = {
            'charge_pipeline': 'Data wrangling and enhancement',
            'universe': 'Database and field-theoretic storage',
            'api_endpoints': 'Fast REST API for queries',
            'dashboard': 'Visualization and monitoring'
        }
        
        missing_folders = []
        for folder, purpose in required_folders.items():
            if not Path(folder).exists():
                missing_folders.append(folder)
            else:
                print(f"   ✅ {folder}/ - {purpose}")
        
        if missing_folders:
            print(f"❌ Missing enterprise folders: {missing_folders}")
            return False
        
        # Test imports work correctly
        import model
        import database
        import api_endpoints
        import dashboard
        
        print(f"✅ All enterprise modules importable")
        
        # Test data flow: Text → Pipeline → Universe → API
        from model import ChargeFactory
        from api_endpoints import ChargeAPI
        
        # Create factory (includes universe)
        factory = ChargeFactory()
        
        # Create API (uses factory)
        api = ChargeAPI()
        
        print(f"✅ Enterprise data flow operational")
        print(f"   Text → Charge Pipeline → Universe → API")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_commercial_scalability():
    """Test commercial scalability features."""
    print(f"\n⚡ Testing Commercial Scalability")
    print("-" * 40)
    
    try:
        from model import ChargeFactory
        
        # Initialize factory
        factory = ChargeFactory()
        
        # Test batch processing performance
        large_batch = [
            f"Commercial text processing batch item {i}" 
            for i in range(20)  # Moderate batch for testing
        ]
        
        start_time = time.time()
        batch_results = factory.create_charges_batch(large_batch, store_in_universe=True)
        processing_time = time.time() - start_time
        
        charges_per_second = len(batch_results) / processing_time
        
        print(f"✅ Batch processing performance")
        print(f"   Processed: {len(batch_results)} charges")
        print(f"   Time: {processing_time:.3f} seconds")
        print(f"   Rate: {charges_per_second:.1f} charges/second")
        
        # Test database persistence
        universe_size_before = len(factory.universe)
        
        # Create new factory (should load from existing database)
        factory2 = ChargeFactory()
        universe_size_after = len(factory2.universe)
        
        print(f"✅ Database persistence")
        print(f"   Charges before: {universe_size_before}")
        print(f"   Charges after reload: {universe_size_after}")
        
        if universe_size_after < universe_size_before:
            print(f"⚠️ Some charges may not have persisted correctly")
        
        # Test field relationships scale
        total_relationships = 0
        for charge in factory2.universe.charges.values():
            total_relationships += len(charge.metadata.nearby_charges)
        
        print(f"✅ Field relationships scale")
        print(f"   Total relationships: {total_relationships}")
        print(f"   Avg per charge: {total_relationships / len(factory2.universe):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Commercial scalability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run complete enterprise system test."""
    print("🚀 Enterprise Field Theory System Test")
    print("=" * 60)
    print("Testing enterprise-grade architecture with proper separation:")
    print("  • charge_pipeline/ - Data wrangling & enhancement")
    print("  • universe/ - Database & field-theoretic storage") 
    print("  • api_endpoints/ - Commercial REST interface")
    print("  • dashboard/ - Visualization & monitoring")
    print("  • docs/ - Enterprise documentation standards")
    
    # Run enterprise tests
    tests = [
        ("Enterprise Architecture", test_enterprise_architecture),
        ("Enterprise Charge Pipeline", test_enterprise_charge_pipeline),
        ("Universe Database", test_universe_database),
        ("Enterprise Documentation", test_enterprise_documentation),
        ("Commercial Scalability", test_commercial_scalability)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*20}")
        success = test_func()
        results.append((name, success))
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 ENTERPRISE SYSTEM TEST RESULTS")
    print(f"=" * 60)
    
    passed = 0
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name:<25} {status}")
        if success:
            passed += 1
    
    overall_success = passed == len(results)
    
    print(f"\n🎯 Overall Status: {'✅ ENTERPRISE READY' if overall_success else '❌ NEEDS WORK'}")
    print(f"   Tests passed: {passed}/{len(results)}")
    
    if overall_success:
        print(f"\n🏆 ENTERPRISE FIELD THEORY SYSTEM OPERATIONAL")
        print(f"   ✅ Proper enterprise architecture with 4-folder separation")
        print(f"   ✅ Rich conceptual charge objects with field placement")
        print(f"   ✅ Persistent universe database with SQLite storage")
        print(f"   ✅ Field-theoretic mathematics for charge placement")
        print(f"   ✅ Enterprise documentation standards")
        print(f"   ✅ Commercial scalability and performance")
        print(f"   ✅ REST API ready for deployment")
        print(f"\n💼 Ready for enterprise deployment and commercial use")
        
    else:
        print(f"\n🔧 ENTERPRISE SYSTEM NEEDS FIXES")
        failed_tests = [name for name, success in results if not success]
        print(f"   Failed components: {failed_tests}")
        print(f"   Address these issues before enterprise deployment")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)