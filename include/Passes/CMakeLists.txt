install(DIRECTORY .
  DESTINATION include/Passes
  FILES_MATCHING
  PATTERN "handlers/*.h"
)

install(DIRECTORY .
  DESTINATION include/Passes
  FILES_MATCHING
  PATTERN "*.h"
)