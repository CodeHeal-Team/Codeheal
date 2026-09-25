// Pre-configured bug scenarios for demo execution.
// Each maps to a "seeded" bug in the sandboxed target repo that
// Part 1 (Core Brain) loads before Part 2's Diagnostic Subagent runs.
//
// `logs` holds the terminal output CodeHeal streams during each phase.
// `code` holds the before/after of the file the Refactoring Subagent patches.
// Once Part 1 + Part 2 are wired up for real, this file goes away and the
// same shape of data streams in live -- see src/lib/backendClient.js.

export const scenarios = [
  {
    id: 'null-pointer',
    label: 'Null Pointer Exception',
    file: 'src/calculator.py',
    testFile: 'src/services/userProfile.bug_reproduce.test.js',
    description: 'Profile lookup crashes when a user has no avatar set.',
    tag: 'Runtime',
    code: {
      before: `function getAvatarUrl(user) {
  return user.avatar.url;
}

module.exports = { getAvatarUrl };`,
      after: `function getAvatarUrl(user) {
  return user.avatar?.url ?? '/default-avatar.png';
}

module.exports = { getAvatarUrl };`,
    },
    logs: {
      analyzing: [
        'Loading src/services/userProfile.js into context...',
        'Scanning call sites for getAvatarUrl()...',
        'Root cause: user.avatar assumed non-null at line 2',
      ],
      testingInitial: [
        '$ npm test -- userProfile',
        'Generated userProfile.bug_reproduce.test.js',
        'FAIL  src/services/userProfile.bug_reproduce.test.js',
        "  \u25cf getAvatarUrl \u203a returns a default when user.avatar is null",
        "    TypeError: Cannot read properties of null (reading 'url')",
        '1 failed, 0 passed',
      ],
      fixing: [
        'Refactoring Subagent: patching src/services/userProfile.js',
        'Adding optional chaining + fallback default',
        'Patch applied (1 file changed, 1 insertion, 1 deletion)',
      ],
      verifying: [
        '$ npm test -- userProfile',
        'PASS  src/services/userProfile.bug_reproduce.test.js',
        '  \u2713 returns a default when user.avatar is null',
        '1 passed, 0 failed',
      ],
    },
  },
  {
    id: 'broken-routing',
    label: 'Broken API Routing',
    file: 'src/routes/orders.js',
    testFile: 'src/routes/orders.bug_reproduce.test.js',
    description: 'PATCH /orders/:id silently matches the wrong route handler.',
    tag: 'API',
    code: {
      before: `router.get('/orders/:id', getOrderHandler);
router.patch('/orders/:id', getOrderHandler);
router.delete('/orders/:id', deleteOrderHandler);`,
      after: `router.get('/orders/:id', getOrderHandler);
router.patch('/orders/:id', updateOrderHandler);
router.delete('/orders/:id', deleteOrderHandler);`,
    },
    logs: {
      analyzing: [
        'Loading src/routes/orders.js into context...',
        'Tracing PATCH /orders/:id request handling...',
        'Root cause: PATCH route bound to getOrderHandler instead of updateOrderHandler',
      ],
      testingInitial: [
        '$ npm test -- orders',
        'Generated orders.bug_reproduce.test.js',
        'FAIL  src/routes/orders.bug_reproduce.test.js',
        '  \u25cf PATCH /orders/:id \u203a persists the updated status',
        '    Expected status 200 with updated body, received unmodified order',
        '1 failed, 0 passed',
      ],
      fixing: [
        'Refactoring Subagent: patching src/routes/orders.js',
        'Rebinding PATCH /orders/:id -> updateOrderHandler',
        'Patch applied (1 file changed, 1 insertion, 1 deletion)',
      ],
      verifying: [
        '$ npm test -- orders',
        'PASS  src/routes/orders.bug_reproduce.test.js',
        '  \u2713 persists the updated status',
        '1 passed, 0 failed',
      ],
    },
  },
  {
    id: 'off-by-one',
    label: 'Off-by-One Math Error',
    file: 'src/utils/pagination.js',
    testFile: 'src/utils/pagination.bug_reproduce.test.js',
    description: 'Last item on each page is dropped from paginated results.',
    tag: 'Logic',
    code: {
      before: `function paginate(items, page, pageSize) {
  const start = page * pageSize;
  const end = start + pageSize - 1;
  return items.slice(start, end);
}`,
      after: `function paginate(items, page, pageSize) {
  const start = page * pageSize;
  const end = start + pageSize;
  return items.slice(start, end);
}`,
    },
    logs: {
      analyzing: [
        'Loading src/utils/pagination.js into context...',
        'Comparing expected vs actual page contents...',
        'Root cause: end index computed one short of the page boundary',
      ],
      testingInitial: [
        '$ npm test -- pagination',
        'Generated pagination.bug_reproduce.test.js',
        'FAIL  src/utils/pagination.bug_reproduce.test.js',
        '  \u25cf paginate \u203a returns a full page of items',
        '    Expected array of length 10, received length 9',
        '1 failed, 0 passed',
      ],
      fixing: [
        'Refactoring Subagent: patching src/utils/pagination.js',
        'Correcting end-index calculation',
        'Patch applied (1 file changed, 1 insertion, 1 deletion)',
      ],
      verifying: [
        '$ npm test -- pagination',
        'PASS  src/utils/pagination.bug_reproduce.test.js',
        '  \u2713 returns a full page of items',
        '1 passed, 0 failed',
      ],
    },
  },
]
