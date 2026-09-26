#ifndef UNIT_TEST_H
#define UNIT_TEST_H

// Reporting for the unit tests: each check first prints what it verifies, then the result, in
// aligned columns. PASS/FAIL are coloured when writing to a terminal (NO_COLOR turns it off).

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <unistd.h>

namespace unit {

inline int checks = 0;
inline int failures = 0;

constexpr int NAME_WIDTH = 36;
constexpr int NOTE_WIDTH = 34;
constexpr int DETAIL_WIDTH = 30;

inline const bool use_color = isatty( STDOUT_FILENO ) && !std::getenv( "NO_COLOR" );

inline std::string paint( const std::string& text, const char* code )
{
   return use_color ? std::string( "\033[" ) + code + "m" + text + "\033[0m" : text;
}

inline std::string green( const std::string& s ) { return paint( s, "32" ); }
inline std::string red( const std::string& s ) { return paint( s, "31" ); }
inline std::string yellow( const std::string& s ) { return paint( s, "33" ); }
inline std::string bold( const std::string& s ) { return paint( s, "1" ); }

// A group of checks; `setup` says what all of them share (sample counts, the measured quantity).
inline void section( const std::string& title, const std::string& setup = "" )
{
   std::cout << '\n' << bold( title );
   if( !setup.empty() )
      std::cout << "  (" << setup << ')';
   std::cout << '\n';
}

// Printed before the computation runs.
inline void announce( const std::string& name, const std::string& note = "" )
{
   const int note_width = std::max< int >( NOTE_WIDTH, note.size() + 2 );
   std::cout << "  " << std::left << std::setw( NAME_WIDTH ) << name << std::setw( note_width ) << note << std::flush;
}

inline bool verdict( bool ok, const std::string& detail = "" )
{
   ++checks;
   failures += !ok;
   std::cout << std::left << std::setw( DETAIL_WIDTH ) << detail << ( ok ? green( "PASS" ) : red( "FAIL" ) ) << '\n';
   return ok;
}

inline std::string measured( double worst, const char* unit )
{
   std::ostringstream s;
   s << "worst " << std::fixed << std::setprecision( 2 ) << std::setw( 5 ) << worst << ' ' << unit;
   return s.str();
}

// Worst measured error against its bound, both in units of `unit`.
inline bool verdict_bound( double worst, double bound, const char* unit = "u^2" )
{
   std::ostringstream s;
   s << measured( worst, unit ) << "  bound " << std::defaultfloat << std::setprecision( 3 ) << bound;
   return verdict( worst <= bound, s.str() );
}

// A measurement without a documented bound: reported, not judged.
inline void info( double worst, const char* unit = "u^2" )
{
   std::cout << std::left << std::setw( DETAIL_WIDTH ) << measured( worst, unit ) + "  no bound" << yellow( "INFO" ) << '\n';
}

inline int finish( const char* name )
{
   std::ostringstream s;
   s << name << ": " << checks - failures << " of " << checks << " checks passed";
   std::cout << '\n' << ( failures == 0 ? green( s.str() ) : red( s.str() ) ) << '\n';
   return failures == 0 ? 0 : 1;
}

}

#endif
