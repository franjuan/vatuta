/**
 * Vatuta library
 */

/**
 * Task
 * @constructor
 * @param {string} Description string for representation of task, no processing over it.
 * @return A new task type
 */
function Task(description) {
	this.description = description;
}
/**
 * Returns task's description
 * @returns {string}
 */
Task.prototype.getDescription = function() {
	return this.description;
}