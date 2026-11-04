"""Training-only wording variation for binding and stateful operations."""

import ast
import re

from dataset.benchmark.src.task_types import CONTRACT
from dataset.generation.src.binding import KEYWORDS


def names(record):
    assignments = [
        node for node in ast.parse(record["setup"]).body if isinstance(node, ast.Assign)
    ]
    return [node.targets[0].id for node in assignments[:2]]


def language_slot(pair):
    return (pair // 12 + pair % 6) % 6


def diversify_binding(record, pair):
    name, other = names(record)
    slot = language_slot(pair)
    if record["family"] == "joined_totals":
        item, group, quantity, container, price = record["binding_axes"]["fields"]
        aggregate = (
            "the sum of quantities"
            if record["variant"]
            else "the sum of quantity multiplied by unit price"
        )
        requests = [
            f"Return {aggregate} per group for {name}. Unit prices live in {other}.",
            f"Using {other} for item pricing, summarize {name} into {aggregate} for each group.",
            f"Reduce the orders in {name} to {aggregate} by group, resolving priced items through {other}.",
            f"Work out {aggregate} separately for every group represented in {name}; the lookup is {other}.",
            f"Please aggregate {aggregate} from {name} by group with item lookup {other}.",
            f"Which groups contribute what amounts? Calculate {aggregate} from order data {name} and pricing object {other}.",
        ]
        roles = (
            f"Order fields are {'inside ' + repr(container) if record['binding_axes']['nested_rows'] else 'at the top level'}: "
            f"{item!r} identifies the item, {group!r} gives its group, and {quantity!r} gives its quantity. "
            f"Prices are {'in field ' + repr(price) + ' of each lookup entry' if record['binding_axes']['nested_prices'] else 'the lookup values'}. "
            "Exclude orders without a lookup match. Ignore unused fields and return a dictionary."
        )
        record["instruction"] = requests[slot] + " " + roles + "\n" + CONTRACT
        record["inspect"] = (
            f"orders = {name}\nprices = {other}\nprint(type(orders).__name__)\nprint(orders)\nprint(type(prices).__name__)\nprint(prices)"
        )
    else:
        call = next(
            node
            for node in ast.walk(ast.parse(record["code"]))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "convert"
        )
        values = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
        }
        multiplier, shift = next(
            options for options in KEYWORDS if set(options) == set(values)
        )
        gain, offset = values[multiplier], values[shift]
        expression = (
            f"(x + {offset}) * {gain}"
            if record["binding_axes"]["order"] == "add_then_multiply"
            else f"x * {gain} + {offset}"
        )
        requests = [
            f"Invoke {other}.convert on {name} for the transformation {expression}. Inspect the accepted keyword names first.",
            f"Transform the numbers {name} through {other}.convert so each x becomes {expression}; discover its call interface.",
            f"Consult {other}.convert's documentation and call that existing method on {name} to obtain {expression} per number.",
            f"The required result is {expression} for every entry of {name}. Use the supplied {other}.convert method with its keyword-only options.",
            f"For the live list {name}, apply {expression} using the actual converter {other}.convert, after reading its help.",
            f"Read the signature of {other}.convert and then use it with {name} to produce the numeric list defined by {expression}.",
        ]
        record["instruction"] = requests[slot] + "\n" + CONTRACT
        record["inspect"] = (
            f"values = {name}\nconverter = {other}\nprint(values)\nhelp(converter.convert)"
        )
    record["id"] = "language-" + record["id"]
    record["contrast_group"] = "language-" + record["contrast_group"]
    record["template_id"] = f"language-{record['family']}-train-{slot}"
    return record


def diversify_support(record, pair):
    family, variant = record["family"], record["variant"]
    slot = language_slot(pair)
    if family == "left_join":
        name, other = names(record)
        rule = (
            "Include only matched orders; discard unknown customers."
            if variant
            else "Include every order, using None for customers absent from the lookup."
        )
        prefixes = [
            "Resolve order customers",
            "Look up customer names",
            "Associate orders with known customer records",
            "Produce customer-enriched order records",
            "Build the ordered customer-name output",
            "Find names for the original order sequence",
        ]
        record["instruction"] = (
            f"{prefixes[slot]} using orders {name} and customers {other} matched by customer_id. "
            f"The customer source column is name; output fields must be order_id and customer_name. {rule} Preserve order sequence.\n"
            + CONTRACT
        )
        aliases = f"orders = {name}\ncustomers = {other}\n"
        if pair % 2:
            guard = " if row['customer_id'] in customer_names" if variant else ""
            record["code"] = (
                aliases
                + "customer_names = dict(zip(customers['customer_id'], customers['name']))\n"
                "result = [{'order_id': int(row['order_id']), 'customer_name': customer_names.get(row['customer_id'])} "
                f"for row in orders.to_dict('records'){guard}]\ndone(str(result))"
            )
        else:
            code = re.sub(r"\b" + re.escape(name) + r"\b", "orders", record["code"])
            code = re.sub(r"\b" + re.escape(other) + r"\b", "customers", code)
            record["code"] = aliases + code
        record["inspect"] = (
            aliases
            + "print(orders)\nprint(customers)\nprint(customers.columns.tolist())"
        )
    elif family == "stateful_followup":
        call = ast.parse(record["followup"]["code"]).body[0].value
        extra, power = call.args[0].left.value, call.args[0].right.value
        requests = [
            f"Extend result in place with the number {extra} raised to exponent {power}.",
            f"Compute power {power} of {extra}, then append that single value to the existing result object.",
            f"Add one more entry to result: {extra} ** {power}. Keep its identity and leave the source alone.",
            f"For exponent {power} and additional input {extra}, append that power to result exactly once.",
            f"Keep the same list result, extending it by one value calculated as {extra} to power {power}.",
            f"The extra base is {extra}. Apply exponent {power} and append the computed value without replacing result.",
        ]
        record["followup"]["instruction"] = (
            requests[slot] + " Preserve the source data."
        )
    record["id"] = "language-" + record["id"]
    record["contrast_group"] = "language-" + record["contrast_group"]
    record["template_id"] = f"language-{family}-train-{slot}"
    return record
