#ifndef UNIT_TEST_H
#define UNIT_TEST_H

// Reporting for the unit tests: each check first prints what it verifies, then the result.

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace unit {

inline int checks = 0;
inline int failures = 0;

inline void section( const std::string& title )
{
   std::cout << '\n' << title << '\n';
}

// Printed before the computation runs.
inline void announce( const std::string& what )
{
   std::cout << "  " << what << " ... " << std::flush;
}

inline bool verdict( bool ok, const std::string& detail = "" )
{
   ++checks;
   failures += !ok;
   if( !detail.empty() )
      std::cout << detail << "  ";
   std::cout << ( ok ? "PASS" : "FAIL" ) << '\n';
   return ok;
}

// Worst measured error against its bound, both in units of u^2 (or whatever `unit` says).
inline bool verdict_bound( double worst, double bound, const char* unit = "u^2" )
{
   std::ostringstream s;
   s << std::setprecision( 3 ) << "worst " << worst << ' ' << unit << " (bound " << bound << ')';
   return verdict( worst <= bound, s.str() );
}

// A measurement without a documented bound: reported, not judged.
inline void info( double worst, const char* unit = "u^2" )
{
   std::cout << std::setprecision( 3 ) << "worst " << worst << ' ' << unit << "  INFO (no documented bound)\n";
}

inline int finish( const char* name )
{
   std::cout << '\n' << name << ": " << checks - failures << " of " << checks << " checks passed\n";
   return failures == 0 ? 0 : 1;
}

}

#endif
